import os, time, shutil, json, random, imagehash, pandas as pd, torch, wandb, matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import local
from io import BytesIO
from base64 import b64encode
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import numpy as np


# 설정
INPUT_CSV_DIR = "/workspace/AIP/miridih-dp-ai-presentation-feature-aippt-v17/datasets/output_data2"
IMAGE_DIR = "/workspace/AIP/Pdf-image-text-search/data/image"
DESCRIPTION_OUT_DIR = Path("description/descriptions_ver5")
PROJECT_NAME = "Messi-blended-v5"

OPENAI_API_KEY = "YOUR-OPENAI-API-KEY"

TEMPERATURE = 0.2
MAX_WORKERS = 10
BATCH_SIZE_EMB = 64
TOP_K = 5

MODE = "csv" 
USE_CSV = False # 기존 CSV 사용 여부 (False면 GPT 생성)

USE_BLEND = True # 임베딩 블렌드 사용 여부
DESC_WEIGHT = 0.6
PAGE_WEIGHT = 0.4


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS  # 하위버전 호환


PROMPT_TEMPLATE = """
# Role
You are an expert visual content analyst. You will describe an image extracted from a PDF. You may use the text from the same PDF page (“page_text”) strictly as background knowledge only if it is clearly and certainly relevant to what is visibly present in the image.

# Background
The purpose of generating this description is to improve text-to-text (T2T) retrieval when matching PDF images to the most relevant presentation slide text.  
We are developing a system that:
1) Takes a PDF as input and generates a presentation based on the PDF's text.
2) Finds the most suitable image from the PDF for each generated slide using T2T similarity search.
Your description will serve as the primary textual representation of the image for retrieval. It must therefore be concise, visually accurate, and only include context that will help match the image to semantically similar slide text.
Additionally, you must determine if the image is visually meaningful and suitable for use in a slide.  
If the image contains only very small, incomplete, or meaningless elements (e.g., a single short word, logo image, cropped text fragments, random shapes, plain background colors, scanning artifacts), mark it as not suitable.

# Context
- The image comes from a PDF page.
- The provided page_text was extracted from that same page but may be unrelated to the image.
- If relevance is uncertain, ignore the page_text entirely.
- Avoid adding any unrelated or speculative details.

# Analysis Process
1) First, examine the image itself and create the most accurate Korean description possible based solely on what is visible.
2) If the image content is unclear or ambiguous, check the page_text for context and only use it if it clearly and certainly relates to the image.
3) If both the image and page_text fail to give certain meaning, describe only the visible form, shape, or general appearance of the image.
4) Do not make predictions or guesses about the meaning, topic, or identity of the image.
5) If the image contains visible text (e.g., a heading, caption, sign), you may use that text content as part of the description.

# Inputs
- page_text: {page_text}
- image: <attached image>

# Task
1) Write a concise Korean description of the image following the above analysis process.
    - For person-centric photos, describe only visible aspects (appearance, pose, background) and do **not** use page_text unless the text is visibly part of the image (e.g., on-screen labels, signs).
    - For non-person images (charts, diagrams, tables, objects, scenes), focus on visible features and use page_text only if it provides certain, unambiguous context that directly matches visible elements (titles, labels, legends, categories).
2) Decide whether the image is suitable for use in a presentation slide (`to_use`):
    - `true` if it is a meaningful, visually clear image that could enhance a slide.
    - `false` if it contains only minimal or cropped text, logo image, meaningless fragments, scanning noise, or is otherwise visually unusable.

# Rules
1) Output MUST be in Korean only for the description.
2) Do not guess or invent details that are not directly visible or certain from the image or page_text.
3) Focus on what is visually present; use page_text only to clarify titles, categories, or relationships that are clearly and directly supported by the image.
4) If page_text and the image conflict, prioritize the image and, if needed, note the likely interpretation without inventing details.
5) Do NOT copy long spans from page_text; summarize only clearly confirmed relevant details.
6) Keep the description concise (exactly one complete sentence), specific, and faithful to the visual content.
7) Do NOT include any additional sections (no “Extracted Text”, no bullet lists, no headers).
8) Under no circumstances output any refusal, apology, or capability disclaimers.

# Output Format
Return the result as valid JSON in the following format:
{{
  "description": "One Korean sentence describing the image, integrating page_text only if it is certain and directly relevant",
  "to_use": true | false
}}
""".strip()

# ======================
# 유틸
# ======================
class PerformanceTracker:
    def __init__(self):
        self.times = dict(dedup=0., description=0., pagetext=0., model_load=0., embedding=0., retrieval=0.)
        self._t0 = None
        self.start = None
        self.end = None
    def start_total(self): self.start = time.time()
    def end_total(self): self.end = time.time()
    def step_start(self): self._t0 = time.time()
    def step_end(self, key): self.times[key] = time.time() - self._t0; self._t0 = None
    def stats(self):
        s = self.times.copy()
        s["total_execution_time"] = (self.end - self.start) if self.start and self.end else 0
        return s

_thread = local()
def _client():
    if not hasattr(_thread, "cli"):
        key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수 또는 상수에 키를 넣어주세요.")
        _thread.cli = OpenAI(api_key=key)
    return _thread.cli

def load_csv_texts(idx):
    # ---- 인덱스 화이트리스트 ----
    allowed = {"1", "11", "24", "31", "36"}
    if str(idx) not in allowed:
        return []

    p = Path(INPUT_CSV_DIR) / f"pdf_text_example_{idx}_image_text.csv"
    if not p.exists():
        return []

    df = None
    last_err = None

    # ---- 인코딩/시그니처 순차 시도 ----
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        # 시도 1: 최신/구버전 공통 파라미터 (onbadlines 없이)
        try:
            df = pd.read_csv(
                p,
                engine="python",
                sep=None,               # 구분자 자동 감지
                quotechar='"',
                doublequote=True,
                skipinitialspace=True,
                encoding=enc,
            )
            break
        except pd.errors.ParserError as e:
            last_err = e
            # 시도 2: 구버전 전용 완화 옵션 (행 스킵)
            try:
                df = pd.read_csv(
                    p,
                    engine="python",
                    sep=None,
                    quotechar='"',
                    doublequote=True,
                    skipinitialspace=True,
                    encoding=enc,
                    error_bad_lines=False,   # pandas<1.3
                    warn_bad_lines=False,
                )
                break
            except Exception as e2:
                last_err = e2
        except Exception as e:
            last_err = e

    if df is None:
        print(f"[ERROR] Failed to read CSV {p}: {last_err}")
        return []

    # ---- 헤더 정규화 & 텍스트 컬럼 선택 ----
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" in df.columns:
        col_text = "text"
    elif "description" in df.columns:
        col_text = "description"
    elif "desc" in df.columns:
        col_text = "desc"
    elif len(df.columns) >= 2:
        col_text = df.columns[1]  # 백업: 두번째 컬럼
    else:
        print(f"[WARN] No usable text column in {p}. Columns={df.columns.tolist()}")
        return []

    # ---- 값 정리 ----
    pairs = []
    for i, t in enumerate(df[col_text].astype(str), 1):
        t = (t or "").strip()
        if not t or t.upper() == "NO IMAGE":
            continue
        pairs.append((f"{p.name}_t{i}", t))
    return pairs



def dedup_folder(folder: Path, tracker, threshold=3):
    tracker.step_start()
    hashes={}; dups=[]
    for f in folder.glob("*.png"):
        try:
            h=imagehash.phash(Image.open(f).convert("RGB"))
            dup = next((hp for hp in hashes if abs(h-hp)<=threshold), None)
            if dup: dups.append(str(f))
            else: hashes[h] = str(f)
        except: pass
    if dups:
        (folder/"duplicate_image").mkdir(exist_ok=True)
        for p in dups: shutil.move(p, folder/"duplicate_image"/Path(p).name)
    tracker.step_end("dedup")

def save_chart(stats, out="perf.png"):
    steps_all = ["description", "pagetext", "model_load", "embedding", "retrieval", "dedup"]
    labels = {"description":"Description","pagetext":"Page Text","model_load":"Model Load","embedding":"Embedding","retrieval":"Retrieval","dedup":"Dedup"}
    steps = [s for s in steps_all if stats.get(s, 0) > 0]
    times = [stats.get(s, 0) for s in steps]
    if not times: return
    total = sum(times); pct = [(t / total) * 100 if total else 0 for t in times]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([labels[s] for s in steps], pct, alpha=0.85, edgecolor="white", linewidth=1.3)
    for bar, p, t in zip(bars, pct, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{p:.1f}%  ({t:.1f}s)", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 100); ax.set_ylabel("Percentage (%)"); ax.set_title("Time Distribution by Stage"); ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout(); Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

def wb_log_csv(run, res_q, descs, tag):
    wandb.init(project=PROJECT_NAME, name=f"{run}_{tag}", reinit=True)
    cols = ["query"] + sum([[f"top-{i+1} image", f"top-{i+1} description"] for i in range(TOP_K)], [])
    tb = wandb.Table(columns=cols)
    for q, lst in res_q.items():
        row = [q]
        for img_path, _, score in lst:
            caption = f"score: {score:.3f}"
            desc = descs.get(img_path, "")
            row += [wandb.Image(img_path, caption=caption), desc]
        while len(row) < len(cols): row.append(None)
        tb.add_data(*row)
    wandb.log({"text_to_image": tb}); wandb.finish()


# page_text 읽기 / GPT 생성
def read_page_text_for_image(img_path: Path, folder: Path) -> str:
    page = next((seg[4:] for seg in img_path.stem.split("_") if seg.startswith("page")), None)
    if not page: return ""
    txt_path = folder / f"text_page{page}.txt"
    if txt_path.exists():
        for enc in ("utf-8","cp949","euc-kr"):
            try:
                return txt_path.read_text(encoding=enc).strip()
            except: pass
    return ""

def _build_messages(image_b64: str, page_text: str):
    prompt = PROMPT_TEMPLATE.format(page_text=page_text[:8000])
    return [{"role": "user", "content": [
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_b64}"}}
    ]}]

def _encode_image_to_b64(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB"); im.thumbnail((512,512), RESAMPLE_LANCZOS)
        buf = BytesIO(); im.save(buf, format="JPEG", quality=90)
    return b64encode(buf.getvalue()).decode()


# 간단 재시도 래퍼 (지수 백오프 + 약간의 지터)
def _retry(fn, *, retries=3, base_delay=0.8):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            sleep_s = base_delay * (2 ** i) + random.uniform(0, 0.2)
            print(f"[WARN] API call failed (attempt {i+1}/{retries}): {e}; retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)


def describe_image_with_context(args):
    path, page_text = args
    b64 = _encode_image_to_b64(path)
    def _do():
        return _client().chat.completions.create(
            model="gpt-4o", temperature=TEMPERATURE, max_tokens=300, 
            messages=_build_messages(b64, page_text),
        ) # model은 gpt-4.1-mini로 변경해서 실험 가능
    try:
        rsp = _retry(_do)
        desc = rsp.choices[0].message.content.strip()
        usage = rsp.usage
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
    except Exception as e:
        print(f"[ERROR] OpenAI call failed for {path}: {e}")
        desc, tin, tout = "", 0, 0
    return str(path), desc, page_text, tin, tout

def generate_descriptions_with_page_text(folder: Path, tracker):
    imgs = sorted([p for p in folder.glob("*.png")])
    jobs = [(p, read_page_text_for_image(p, folder)) for p in imgs]
    in_tok = out_tok = 0

    raw_descs: Dict[str, str] = {}
    page_text_map_all: Dict[str, str] = {}

    tracker.step_start()
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for p, d, pt, tin, tout in tqdm(
            ex.map(describe_image_with_context, jobs),
            total=len(jobs),
            desc="GPT 설명(page_text 포함)"
        ):
            raw_descs[p] = d
            page_text_map_all[p] = pt
            in_tok += tin
            out_tok += tout
    tracker.step_end("description")

    # 코드펜스/라벨까지 대비한 JSON 추출
    def _extract_json(s: str) -> dict:
        s = (s or "").strip()
        if s.startswith("```"):
            s = s.strip().strip("`")
            if s.lower().startswith("json"):
                s = s[4:].lstrip("\n\r\t :")
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            s = s[l:r+1]
        try:
            return json.loads(s)
        except Exception:
            return {}

    # 검색 갤러리에 쓸 채택 항목 (to_use=True & desc 존재)
    descs: Dict[str, str] = {}
    page_text_map: Dict[str, str] = {}

    # CSV에 저장할 모든 항목 (True/False 모두)
    all_image_paths: List[str] = []
    all_descriptions: List[str] = []
    all_to_use: List[bool] = []

    skipped_count = 0
    bad_samples = []

    for p, raw in raw_descs.items():
        obj = _extract_json(raw)
        to_use = bool(obj.get("to_use", False))
        desc = obj.get("description")
        # CSV용 전체 기록
        all_image_paths.append(p)
        all_descriptions.append(desc.strip() if isinstance(desc, str) else "")
        all_to_use.append(to_use)

        # 검색용 선별
        if to_use and isinstance(desc, str) and desc.strip():
            descs[p] = desc.strip()
            page_text_map[p] = page_text_map_all.get(p, "")
        else:
            skipped_count += 1
            bad_samples.append(p)

    # 비용 통계
    cost = in_tok * 2.5 / 1e6 + out_tok * 10 / 1e6
    avg = cost / max(len(raw_descs), 1)

    # === CSV 저장: True/False 모두 포함 ===
    DESCRIPTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DESCRIPTION_OUT_DIR / f"{folder.name}.csv"
    pd.DataFrame({
        "image_path": all_image_paths,
        "description": all_descriptions,
        "to_use": all_to_use
    }).to_csv(out_path, index=False, encoding="utf-8-sig")

    if skipped_count:
        print(f"[INFO] {skipped_count} images skipped for retrieval (to_use=false or invalid JSON). Examples: {bad_samples[:5]}")

    return descs, page_text_map, cost, avg


# ===== Helper: L2 정규화 =====
def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        denom = max(np.linalg.norm(x), eps)
        return x / denom
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        return x / norms

# ===== NEW: 쿼리 필터링 (공백 제외 10자 이하 스킵) =====
def filter_queries(queries: List[str], min_len_no_space: int = 11) -> Tuple[List[str], List[str]]:
    """
    공백(스페이스/탭/개행 등)을 모두 제거한 길이가 min_len_no_space 미만이면 스킵.
    기본값 11 => '공백 제외 10글자 이하' 스킵.
    """
    kept, skipped = [], []
    for q in queries:
        q_no_space = "".join(q.split())
        if len(q_no_space) < min_len_no_space:
            skipped.append(q)
        else:
            kept.append(q)
    return kept, skipped


# ======================
# 메인 파이프라인
# ======================
def main():
    tracker = PerformanceTracker(); tracker.start_total()
    print("CUDA Available:", torch.cuda.is_available())
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    tracker.step_start()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
    try: model.max_seq_length = 512
    except: pass
    tracker.step_end("model_load")

    allowed = {"1", "11", "24", "31", "36"}
    folders = sorted([
        p for p in Path(IMAGE_DIR).iterdir()
        if p.is_dir()
        and p.name.endswith("_single")
        and p.name.split("_")[0] in allowed
    ])

    for folder in folders:
        run = folder.name
        idx = run.split("_")[0]
        csv_texts = load_csv_texts(idx)
        if not csv_texts:
            print(f"[INFO] No CSV texts for idx={idx}, skip {run}")
            continue

        dedup_folder(folder, tracker)

        if USE_CSV:
            # 기존 Description CSV 사용 (이미 to_use=true로 필터된 CSV를 사용한다고 가정)
            desc_file = DESCRIPTION_OUT_DIR / f"{folder.name}.csv"
            if not desc_file.exists():
                print(f"[WARN] Description CSV not found: {desc_file}, skip {run}")
                continue
            df_desc = pd.read_csv(desc_file)

            # <-- 추가: to_use 컬럼이 있으면 True만 유지
            if "to_use" in df_desc.columns:
                before = len(df_desc)
                df_desc = df_desc[df_desc["to_use"] == True].copy()
                after = len(df_desc)
                if after < before:
                    print(f"[INFO] Filtered by to_use=True: {before} -> {after}")

            if "image_path" not in df_desc.columns or "description" not in df_desc.columns:
                print(f"[WARN] Invalid CSV format in: {desc_file}, skip {run}")
                continue

            descs = dict(zip(df_desc["image_path"], df_desc["description"]))
            # page_text는 해당 이미지별로 다시 로드
            page_texts = {p: read_page_text_for_image(Path(p), folder) for p in descs.keys()}
            cost, avg = 0, 0
        else:
            descs, page_texts, cost, avg = generate_descriptions_with_page_text(folder, tracker)

        # === to_use=true가 0개면 스킵 ===
        if not descs:
            print(f"[WARN] No images marked to_use=true for {run}; skip retrieval for this folder.")
            tracker.step_start(); tracker.step_end("embedding")
            tracker.step_start(); tracker.step_end("retrieval")
            tracker.end_total()
            stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": 0}
            save_chart(stats, f"perf_{run}.png")
            wb_log_csv(run, {}, {}, "descOnly")
            continue

        # ===== 임베딩 =====
        tracker.step_start()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        desc_list = list(descs.values())
        desc_emb = model.encode(desc_list, normalize_embeddings=True,
                                batch_size=BATCH_SIZE_EMB, show_progress_bar=True)

        if USE_BLEND and PAGE_WEIGHT > 0:
            pt_list = [page_texts[p] for p in descs.keys()]
            text_emb = model.encode(pt_list, normalize_embeddings=True,
                                    batch_size=BATCH_SIZE_EMB, show_progress_bar=True)
            blended = DESC_WEIGHT * desc_emb + PAGE_WEIGHT * text_emb
            blended = l2norm(blended)
            gallery_emb = blended
            tag = f"blend_{DESC_WEIGHT:.2f}:{PAGE_WEIGHT:.2f}"
        else:
            gallery_emb = desc_emb
            tag = "descOnly"

        # ===== 쿼리 필터링 =====
        q_txt_raw = [t for _, t in csv_texts]
        q_txt, q_skipped = filter_queries(q_txt_raw, min_len_no_space=11)

        if q_skipped:
            print(f"[INFO] {len(q_skipped)} queries skipped (<=10 chars without spaces). Examples:")
            for s in q_skipped[:5]:
                print("   -", repr(s))

        if not q_txt:
            print(f"[WARN] All queries skipped for {run}; skip retrieval for this folder.")
            tracker.step_end("embedding")
            tracker.step_start(); tracker.step_end("retrieval")
            tracker.end_total()
            stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": len(descs)}
            save_chart(stats, f"perf_{run}.png")
            wb_log_csv(run, {}, descs, tag)
            continue

        q_emb = model.encode(q_txt, normalize_embeddings=True, batch_size=BATCH_SIZE_EMB, show_progress_bar=True)

        gallery_t = torch.tensor(gallery_emb, dtype=torch.float32).to(device)
        q_t = torch.tensor(q_emb, dtype=torch.float32).to(device)
        tracker.step_end("embedding")

        # Retrieval
        tracker.step_start()
        img_paths_order = list(descs.keys())  # descs는 삽입순서를 유지(dict in Py3.7+)
        res_q = {}
        for q, qe in zip(q_txt, q_t):
            sc = util.cos_sim(qe, gallery_t)[0]
            top = torch.topk(sc, k=min(TOP_K, len(sc))).indices.tolist()
            res_q[q] = [(img_paths_order[i], "", sc[i].item()) for i in top]
        tracker.step_end("retrieval")

        # 요약/로그/종료
        tracker.end_total()
        stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": len(descs)}
        save_chart(stats, f"perf_{run}.png")
        wb_log_csv(run, res_q, descs, tag)

if __name__ == "__main__":
    main()





'''
GPT-5 버전 실행 (https://miridih.atlassian.net/wiki/spaces/~712020c4a8435f9991491d970b32b8a680aad5/pages/1262747711/4.+R+D -> 해당 위키파일 참고)
'''
# import os, time, shutil, json, random, imagehash, pandas as pd, torch, wandb, matplotlib.pyplot as plt
# from typing import List, Tuple, Dict
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
# from threading import local
# from io import BytesIO
# from base64 import b64encode
# from tqdm import tqdm
# from PIL import Image
# from sentence_transformers import SentenceTransformer, util
# from openai import OpenAI
# import numpy as np

# # 설정
# INPUT_CSV_DIR = "/workspace/AIP/miridih-dp-ai-presentation-feature-aippt-v17/datasets/output_data2"
# IMAGE_DIR = "/workspace/AIP/Pdf-image-text-search/data/image"
# DESCRIPTION_OUT_DIR = Path("description/descriptions_ver7(gpt-5)")
# PROJECT_NAME = "Messi-blended-v7"

# OPENAI_API_KEY = "YOUR-OPENAI-API-KEY"


# MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5")
# PRICES = {
#     "gpt-5": (1.25, 10.0),
#     "gpt-5-mini": (0.25, 2.0),
#     "gpt-5-nano": (0.05, 0.40),
#     "gpt-5-chat-latest": (1.25, 10.0),
# }

# MAX_WORKERS = 10
# BATCH_SIZE_EMB = 64
# TOP_K = 5

# MODE = "csv"
# USE_CSV = False

# USE_BLEND = True
# DESC_WEIGHT = 0.6
# PAGE_WEIGHT = 0.4

# # Pillow LANCZOS 호환 (Pillow>=10)
# try:
#     RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
# except AttributeError:
#     RESAMPLE_LANCZOS = Image.LANCZOS  # 하위버전 호환

# # ===== 프롬프트 (page_text 포함) =====
# PROMPT_TEMPLATE = """
# # Role
# You are an expert visual content analyst. You will describe an image extracted from a PDF. You may use the text from the same PDF page (“page_text”) strictly as background knowledge only if it is clearly and certainly relevant to what is visibly present in the image.

# # Background
# The purpose of generating this description is to improve text-to-text (T2T) retrieval when matching PDF images to the most relevant presentation slide text.  
# We are developing a system that:
# 1) Takes a PDF as input and generates a presentation based on the PDF's text.
# 2) Finds the most suitable image from the PDF for each generated slide using T2T similarity search.
# Your description will serve as the primary textual representation of the image for retrieval. It must therefore be concise, visually accurate, and only include context that will help match the image to semantically similar slide text.
# Additionally, you must determine if the image is visually meaningful and suitable for use in a slide.  
# If the image contains only very small, incomplete, or meaningless elements (e.g., a single short word, logo image, cropped text fragments, random shapes, plain background colors, scanning artifacts), mark it as not suitable.

# # Context
# - The image comes from a PDF page.
# - The provided page_text was extracted from that same page but may be unrelated to the image.
# - If relevance is uncertain, ignore the page_text entirely.
# - Avoid adding any unrelated or speculative details.

# # Analysis Process
# 1) First, examine the image itself and create the most accurate Korean description possible based solely on what is visible.
# 2) If the image content is unclear or ambiguous, check the page_text for context and only use it if it clearly and certainly relates to the image.
# 3) If both the image and page_text fail to give certain meaning, describe only the visible form, shape, or general appearance of the image.
# 4) Do not make predictions or guesses about the meaning, topic, or identity of the image.
# 5) If the image contains visible text (e.g., a heading, caption, sign), you may use that text content as part of the description.

# # Inputs
# - page_text: {page_text}
# - image: <attached image>

# # Task
# 1) Write a concise Korean description of the image following the above analysis process.
#     - For person-centric photos, describe only visible aspects (appearance, pose, background) and do **not** use page_text unless the text is visibly part of the image (e.g., on-screen labels, signs).
#     - For non-person images (charts, diagrams, tables, objects, scenes), focus on visible features and use page_text only if it provides certain, unambiguous context that directly matches visible elements (titles, labels, legends, categories).
# 2) Decide whether the image is suitable for use in a presentation slide (`to_use`):
#     - `true` if it is a meaningful, visually clear image that could enhance a slide.
#     - `false` if it contains only minimal or cropped text, logo image, meaningless fragments, scanning noise, or is otherwise visually unusable.

# # Rules
# 1) Output MUST be in Korean only for the description.
# 2) Do not guess or invent details that are not directly visible or certain from the image or page_text.
# 3) Focus on what is visually present; use page_text only to clarify titles, categories, or relationships that are clearly and directly supported by the image.
# 4) If page_text and the image conflict, prioritize the image and, if needed, note the likely interpretation without inventing details.
# 5) Do NOT copy long spans from page_text; summarize only clearly confirmed relevant details.
# 6) Keep the description concise (exactly one complete sentence), specific, and faithful to the visual content.
# 7) Do NOT include any additional sections (no “Extracted Text”, no bullet lists, no headers).
# 8) Under no circumstances output any refusal, apology, or capability disclaimers.

# # Output Format
# Return the result as valid JSON in the following format:
# {{
#     "description": "One Korean sentence describing the image, integrating page_text only if it is certain and directly relevant",
#     "to_use": true | false
# }}
# """.strip()


# # 유틸
# class PerformanceTracker:
#     def __init__(self):
#         self.times = dict(dedup=0., description=0., model_load=0., embedding=0., retrieval=0.)
#         self._t0 = None
#         self.start = None
#         self.end = None
#     def start_total(self): self.start = time.time()
#     def end_total(self): self.end = time.time()
#     def step_start(self): self._t0 = time.time()
#     def step_end(self, key): self.times[key] = time.time() - self._t0; self._t0 = None
#     def stats(self):
#         s = self.times.copy()
#         s["total_execution_time"] = (self.end - self.start) if self.start and self.end else 0
#         return s


# _thread = local()
# def _client():
#     if not hasattr(_thread, "cli"):
#         key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
#         if not key:
#             raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수 또는 상수에 키를 넣어주세요.")
#         _thread.cli = OpenAI(api_key=key)
#     return _thread.cli


# def load_csv_texts(idx):
#     # ---- 인덱스 화이트리스트 ----
#     allowed = {"1", "11", "24", "31", "36"}
#     if str(idx) not in allowed:
#         return []

#     p = Path(INPUT_CSV_DIR) / f"pdf_text_example_{idx}_image_text.csv"
#     if not p.exists():
#         return []

#     df = None
#     last_err = None

#     for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
#         try:
#             df = pd.read_csv(
#                 p, engine="python", sep=None, quotechar='"',
#                 doublequote=True, skipinitialspace=True, encoding=enc,
#             ); break
#         except pd.errors.ParserError as e:
#             last_err = e
#             try:
#                 df = pd.read_csv(
#                     p, engine="python", sep=None, quotechar='"',
#                     doublequote=True, skipinitialspace=True, encoding=enc,
#                     error_bad_lines=False, warn_bad_lines=False,
#                 ); break
#             except Exception as e2:
#                 last_err = e2
#         except Exception as e:
#             last_err = e

#     if df is None:
#         print(f"[ERROR] Failed to read CSV {p}: {last_err}"); return []

#     df.columns = [str(c).strip().lower() for c in df.columns]
#     if "text" in df.columns: col_text = "text"
#     elif "description" in df.columns: col_text = "description"
#     elif "desc" in df.columns: col_text = "desc"
#     elif len(df.columns) >= 2: col_text = df.columns[1]
#     else:
#         print(f"[WARN] No usable text column in {p}. Columns={df.columns.tolist()}"); return []

#     pairs=[]
#     for i, t in enumerate(df[col_text].astype(str), 1):
#         t=(t or "").strip()
#         if not t or t.upper()=="NO IMAGE": continue
#         pairs.append((f"{p.name}_t{i}", t))
#     return pairs


# def dedup_folder(folder: Path, tracker, threshold=3):
#     tracker.step_start()
#     hashes={}; dups=[]
#     for f in folder.glob("*.png"):
#         try:
#             h=imagehash.phash(Image.open(f).convert("RGB"))
#             dup = next((hp for hp in hashes if abs(h-hp)<=threshold), None)
#             if dup: dups.append(str(f))
#             else: hashes[h] = str(f)
#         except: pass
#     if dups:
#         (folder/"duplicate_image").mkdir(exist_ok=True)
#         for p in dups: shutil.move(p, folder/"duplicate_image"/Path(p).name)
#     tracker.step_end("dedup")


# def save_chart(stats, out="perf.png"):
#     steps_all = ["description", "model_load", "embedding", "retrieval", "dedup"]
#     labels = {"description":"Description","model_load":"Model Load","embedding":"Embedding","retrieval":"Retrieval","dedup":"Dedup"}
#     steps = [s for s in steps_all if stats.get(s, 0) > 0]
#     times = [stats.get(s, 0) for s in steps]

#     if not times: return

#     total = sum(times); pct = [(t / total) * 100 if total else 0 for t in times]
#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars = ax.bar([labels[s] for s in steps], pct, alpha=0.85, edgecolor="white", linewidth=1.3)

#     for bar, p, t in zip(bars, pct, times):
#         ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{p:.1f}%  ({t:.1f}s)", ha="center", va="bottom", fontweight="bold")
#     ax.set_ylim(0, 100); ax.set_ylabel("Percentage (%)"); ax.set_title("Time Distribution by Stage"); ax.grid(axis="y", alpha=0.3, linestyle="--")
#     fig.tight_layout(); Path(out).parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


# def wb_log_csv(run, res_q, descs, tag):
#     wandb.init(project=PROJECT_NAME, name=f"{run}_{tag}", reinit=True)
#     cols = ["query"] + sum([[f"top-{i+1} image", f"top-{i+1} description"] for i in range(TOP_K)], [])
#     tb = wandb.Table(columns=cols)
#     for q, lst in res_q.items():
#         row = [q]
#         for img_path, _, score in lst:
#             caption = f"score: {score:.3f}"
#             desc = descs.get(img_path, "")
#             row += [wandb.Image(img_path, caption=caption), desc]
#         while len(row) < len(cols): row.append(None)
#         tb.add_data(*row)
#     wandb.log({"text_to_image": tb}); wandb.finish()

# # ======================
# # page_text 읽기 (프롬프트/블렌딩 둘 다 사용)
# # ======================
# def read_page_text_for_image(img_path: Path, folder: Path) -> str:
#     page = next((seg[4:] for seg in img_path.stem.split("_") if seg.startswith("page")), None)
#     if not page: return ""
#     txt_path = folder / f"text_page{page}.txt"
#     if txt_path.exists():
#         for enc in ("utf-8","cp949","euc-kr"):
#             try:
#                 return txt_path.read_text(encoding=enc).strip()
#             except: pass
#     return ""


# def _extract_json(s: str) -> dict:
#     s = (s or "").strip()
#     if s.startswith("```"):
#         s = s.strip().strip("`")
#         if s.lower().startswith("json"):
#             s = s[4:].lstrip("\n\r\t :")
#     l = s.find("{"); r = s.rfind("}")
#     if l != -1 and r != -1 and r > l:
#         s = s[l:r+1]
#     try:
#         return json.loads(s)
#     except Exception:
#         return {}


# # Responses API 입력 구성 (멀티모달) — page_text 주입
# def _build_responses_input(image_b64: str, page_text: str):
#     prompt = PROMPT_TEMPLATE.format(page_text=(page_text or "")[:8000])
#     return [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": prompt},
#                 {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
#             ],
#         }
#     ]

# # Chat Completions용 메시지(폴백) — page_text 주입
# def _build_chat_messages(image_b64: str, page_text: str):
#     prompt = PROMPT_TEMPLATE.format(page_text=(page_text or "")[:8000])
#     return [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
#         ],
#     }]

# def _encode_image_to_b64(path: Path) -> str:
#     with Image.open(path) as im:
#         im = im.convert("RGB"); im.thumbnail((512,512), RESAMPLE_LANCZOS)
#         buf = BytesIO(); im.save(buf, format="JPEG", quality=90)
#     return b64encode(buf.getvalue()).decode()

# # 간단 재시도 래퍼 (지수 백오프 + 약간의 지터)
# def _retry(fn, *, retries=3, base_delay=0.8):
#     for i in range(retries):
#         try:
#             return fn()
#         except Exception as e:
#             if i == retries - 1:
#                 raise
#             sleep_s = base_delay * (2 ** i) + random.uniform(0, 0.2)
#             print(f"[WARN] API call failed (attempt {i+1}/{retries}): {e}; retry in {sleep_s:.1f}s")
#             time.sleep(sleep_s)

# def describe_image(args):
#     path, page_text = args
#     b64 = _encode_image_to_b64(path)
#     cli = _client()

#     def _do():
#         if MODEL_NAME.startswith("gpt-5"):
#             effort = os.getenv("GPT5_REASONING_EFFORT", "low")
#             verbosity = os.getenv("GPT5_VERBOSITY", "low")
#             return cli.responses.create(
#                 model=MODEL_NAME,
#                 input=_build_responses_input(b64, page_text),
#                 reasoning={"effort": effort},
#                 text={"verbosity": verbosity},
#             )
#         else:
#             return cli.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=_build_chat_messages(b64, page_text),
#             )

#     try:
#         rsp = _retry(_do)
#         if hasattr(rsp, "output_text"):
#             desc = (rsp.output_text or "").strip()
#         else:
#             if not rsp or not getattr(rsp, "choices", None):
#                 raise RuntimeError("Empty response")
#             desc = rsp.choices[0].message.content.strip()

#         usage = getattr(rsp, "usage", None)
#         if usage and hasattr(usage, "input_tokens"):
#             tin  = getattr(usage, "input_tokens", 0)
#             tout = getattr(usage, "output_tokens", 0)
#         else:
#             usage2 = getattr(rsp, "usage", None)
#             tin  = getattr(usage2, "prompt_tokens", 0) if usage2 else 0
#             tout = getattr(usage2, "completion_tokens", 0) if usage2 else 0

#     except Exception as e:
#         print(f"[ERROR] OpenAI call failed for {path}: {e}")
#         desc, tin, tout = "", 0, 0
#     return str(path), desc, tin, tout

# def generate_descriptions(folder: Path, tracker):
#     imgs = sorted([p for p in folder.glob("*.png")])
#     in_tok = out_tok = 0

#     # 블렌딩 & 프롬프트용 page_text 미리 수집
#     page_text_map_all: Dict[str, str] = {str(p): read_page_text_for_image(p, folder) for p in imgs}

#     # ---- GPT 호출 (page_text 포함) ----
#     raw_resp: Dict[str, str] = {}  # image_path(str) -> GPT 원문 응답(그대로 CSV 저장)
#     jobs = [(p, page_text_map_all[str(p)]) for p in imgs]

#     tracker.step_start()
#     with ThreadPoolExecutor(MAX_WORKERS) as ex:
#         for p, d, tin, tout in tqdm(
#             ex.map(describe_image, jobs),
#             total=len(jobs),
#             desc="GPT 설명(page_text 포함)"
#         ):
#             raw_resp[p] = d
#             in_tok += tin
#             out_tok += tout
#     tracker.step_end("description")

#     # ---- CSV 저장(원문 그대로) ----
#     DESCRIPTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
#     out_path = DESCRIPTION_OUT_DIR / f"{folder.name}.csv"
#     paths_in_order = [str(p) for p in imgs]
#     descs_in_order = [raw_resp.get(str(p), "") for p in imgs]
#     pd.DataFrame({"image_path": paths_in_order, "description": descs_in_order}).to_csv(
#         out_path, index=False, encoding="utf-8-sig"
#     )

#     # ---- 검색/임베딩용 파싱: JSON에서 to_use==true만 선택, description만 사용 ----
#     descs: Dict[str, str] = {}
#     page_texts: Dict[str, str] = {}
#     skipped = []

#     for p, raw in raw_resp.items():
#         obj = _extract_json(raw)
#         to_use = bool(obj.get("to_use", False))
#         desc_text = obj.get("description")
#         if to_use and isinstance(desc_text, str) and desc_text.strip():
#             descs[p] = desc_text.strip()
#             page_texts[p] = page_text_map_all.get(p, "")
#         else:
#             skipped.append(p)

#     # ---- 비용 추정 (usage 있을 경우만) ----
#     in_rate, out_rate = PRICES.get(MODEL_NAME, (1.25, 10.0))
#     cost = in_tok * in_rate / 1e6 + out_tok * out_rate / 1e6 if (in_tok or out_tok) else 0.0
#     avg = cost / max(len(raw_resp), 1) if cost else 0.0

#     if skipped:
#         print(f"[INFO] {len(skipped)} images excluded from retrieval (to_use=false or invalid JSON). Examples: {skipped[:5]}")

#     return descs, page_texts, cost, avg

# # ===== Helper: L2 정규화 =====
# def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     if x.ndim == 1:
#         denom = max(np.linalg.norm(x), eps)
#         return x / denom
#     else:
#         norms = np.linalg.norm(x, axis=1, keepdims=True)
#         norms = np.clip(norms, eps, None)
#         return x / norms

# # ===== NEW: 쿼리 필터링 (공백 제외 10자 이하 스킵) =====
# def filter_queries(queries: List[str], min_len_no_space: int = 11) -> Tuple[List[str], List[str]]:
#     kept, skipped = [], []
#     for q in queries:
#         q_no_space = "".join(q.split())
#         if len(q_no_space) < min_len_no_space:
#             skipped.append(q)
#         else:
#             kept.append(q)
#     return kept, skipped

# # ======================
# # 메인 파이프라인
# # ======================
# def main():
#     tracker = PerformanceTracker(); tracker.start_total()
#     print("CUDA Available:", torch.cuda.is_available())
#     print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

#     # 모델 로드
#     tracker.step_start()
#     model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
#     try: model.max_seq_length = 512
#     except: pass
#     tracker.step_end("model_load")

#     allowed = {"1", "11", "24", "31", "36"}
#     folders = sorted([
#         p for p in Path(IMAGE_DIR).iterdir()
#         if p.is_dir()
#         and p.name.endswith("_single")
#         and p.name.split("_")[0] in allowed
#     ])

#     for folder in folders:
#         run = folder.name
#         idx = run.split("_")[0]
#         csv_texts = load_csv_texts(idx)
#         if not csv_texts:
#             print(f"[INFO] No CSV texts for idx={idx}, skip {run}")
#             continue

#         dedup_folder(folder, tracker)

#         if USE_CSV:
#             # CSV에는 "원문 응답"이 들어 있음 -> 여기서 JSON 파싱하여 to_use==true만 선택
#             desc_file = DESCRIPTION_OUT_DIR / f"{folder.name}.csv"
#             if not desc_file.exists():
#                 print(f"[WARN] Description CSV not found: {desc_file}, skip {run}")
#                 continue
#             df_desc = pd.read_csv(desc_file)
#             if "image_path" not in df_desc.columns or "description" not in df_desc.columns:
#                 print(f"[WARN] Invalid CSV format in: {desc_file}, skip {run}")
#                 continue

#             descs: Dict[str, str] = {}
#             page_texts: Dict[str, str] = {}
#             skipped_rows = 0

#             for _, row in df_desc.iterrows():
#                 p = str(row["image_path"])
#                 raw = str(row["description"]) if not pd.isna(row["description"]) else ""
#                 obj = _extract_json(raw)
#                 to_use = bool(obj.get("to_use", False))
#                 desc_text = obj.get("description")
#                 if to_use and isinstance(desc_text, str) and desc_text.strip():
#                     descs[p] = desc_text.strip()
#                     page_texts[p] = read_page_text_for_image(Path(p), folder)
#                 else:
#                     skipped_rows += 1

#             if skipped_rows:
#                 print(f"[INFO] {skipped_rows} images excluded from retrieval by CSV (to_use=false/invalid).")

#             cost, avg = 0, 0  # CSV 경로에선 usage/cost 없음

#         else:
#             # 새로 GPT 호출 -> CSV에 원문 저장, 임베딩용은 JSON 파싱해 필터링
#             descs, page_texts, cost, avg = generate_descriptions(folder, tracker)

#         # === 텍스트가 없는 경우 스킵 ===
#         if not descs:
#             print(f"[WARN] No descriptions with to_use=true for {run}; skip retrieval for this folder.")
#             tracker.step_start(); tracker.step_end("embedding")
#             tracker.step_start(); tracker.step_end("retrieval")
#             tracker.end_total()
#             stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": 0}
#             save_chart(stats, f"perf_{run}.png")
#             wb_log_csv(run, {}, {}, "descOnly")
#             continue

#         # ===== 임베딩 =====
#         tracker.step_start()
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         img_keys = list(descs.keys())

#         # 1) Description 임베딩 (JSON에서 뽑은 description만)
#         desc_list = [descs[k] for k in img_keys]
#         desc_emb = model.encode(
#             desc_list,
#             normalize_embeddings=True,
#             batch_size=BATCH_SIZE_EMB,
#             show_progress_bar=True
#         )

#         # 2) Page text 임베딩(블렌딩용)
#         if USE_BLEND and PAGE_WEIGHT > 0:
#             pt_list = [page_texts.get(k, "") for k in img_keys]
#             text_emb = model.encode(
#                 pt_list,
#                 normalize_embeddings=True,
#                 batch_size=BATCH_SIZE_EMB,
#                 show_progress_bar=True
#             )
#             blended = DESC_WEIGHT * desc_emb + PAGE_WEIGHT * text_emb
#             gallery_emb = l2norm(blended)
#             tag = f"blend_{DESC_WEIGHT:.2f}:{PAGE_WEIGHT:.2f}"
#         else:
#             gallery_emb = desc_emb
#             tag = "descOnly"

#         # ===== 쿼리 필터링 =====
#         q_txt_raw = [t for _, t in csv_texts]
#         q_txt, q_skipped = filter_queries(q_txt_raw, min_len_no_space=11)

#         if q_skipped:
#             print(f"[INFO] {len(q_skipped)} queries skipped (<=10 chars without spaces). Examples:")
#             for s in q_skipped[:5]:
#                 print("   -", repr(s))

#         if not q_txt:
#             print(f"[WARN] All queries skipped for {run}; skip retrieval for this folder.")
#             tracker.step_end("embedding")
#             tracker.step_start(); tracker.step_end("retrieval")
#             tracker.end_total()
#             stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": len(descs)}
#             save_chart(stats, f"perf_{run}.png")
#             wb_log_csv(run, {}, descs, tag)
#             continue

#         q_emb = model.encode(q_txt, normalize_embeddings=True,
#                              batch_size=BATCH_SIZE_EMB, show_progress_bar=True)

#         gallery_t = torch.tensor(gallery_emb, dtype=torch.float32).to(device)
#         q_t = torch.tensor(q_emb, dtype=torch.float32).to(device)
#         tracker.step_end("embedding")

#         # Retrieval
#         tracker.step_start()
#         img_paths_order = img_keys
#         res_q = {}
#         for q, qe in zip(q_txt, q_t):
#             sc = util.cos_sim(qe, gallery_t)[0]
#             top = torch.topk(sc, k=min(TOP_K, len(sc))).indices.tolist()
#             res_q[q] = [(img_paths_order[i], "", sc[i].item()) for i in top]
#         tracker.step_end("retrieval")

#         # 요약/로그/종료
#         tracker.end_total()
#         stats = tracker.stats() | {"total_cost": cost, "avg_cost": avg, "samples": len(descs)}
#         save_chart(stats, f"perf_{run}.png")
#         wb_log_csv(run, res_q, descs, tag)

# if __name__ == "__main__":
#     main()
