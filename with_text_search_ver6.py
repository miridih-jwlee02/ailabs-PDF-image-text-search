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
DESCRIPTION_OUT_DIR = Path("description/descriptions_ver6")
PROJECT_NAME = "Messi-blended-v6"

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
You are an expert visual content analyst. You will describe an image extracted from a PDF.

# Background
The purpose of generating this description is to improve text-to-text (T2T) retrieval when matching PDF images to the most relevant presentation slide text.
Your description will serve as the primary textual representation of the image for retrieval. It must therefore be concise, visually accurate, and focused on what is visibly present.
Additionally, you must determine if the image is visually meaningful and suitable for use in a slide.
If the image contains only very small, incomplete, or meaningless elements (e.g., a single short word, logo image, cropped text fragments, random shapes, plain background colors, scanning artifacts), mark it as not suitable.

# Context
- The image comes from a PDF page.
- Avoid adding any unrelated or speculative details.

# Analysis Process
1) First, examine the image itself and create the most accurate Korean description possible based solely on what is visible.
2) Do not make predictions or guesses about the meaning, topic, or identity of the image.
3) If the image contains visible text (e.g., a heading, caption, sign), you may use that text content as part of the description.

# Inputs
- image: <attached image>

# Task
1) Write a concise Korean description of the image following the above analysis process.
   - For person-centric photos, describe only visible aspects (appearance, pose, background).
   - For non-person images (charts, diagrams, tables, objects, scenes), focus on visible features such as titles, labels, legends, axes, shapes, and layout that are clearly seen.
2) Decide whether the image is suitable for use in a presentation slide (`to_use`):
   - `true` if it is a meaningful, visually clear image that could enhance a slide.
   - `false` if it contains only minimal or cropped text, logo image, meaningless fragments, scanning noise, or is otherwise visually unusable.

# Rules
1) Output MUST be in Korean only for the description.
2) Do not guess or invent details that are not directly visible.
3) Keep the description concise (exactly one complete sentence), specific, and faithful to the visual content.
4) Do NOT include any additional sections (no “Extracted Text”, no bullet lists, no headers).
5) Under no circumstances output any refusal, apology, or capability disclaimers.

# Output Format
Return the result as valid JSON in the following format:
{
  "description": "One Korean sentence describing the image",
  "to_use": true | false
}
""".strip()


# 유틸
class PerformanceTracker:
    def __init__(self):
        self.times = dict(dedup=0., description=0., model_load=0., embedding=0., retrieval=0.)
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
    steps_all = ["description", "model_load", "embedding", "retrieval", "dedup"]
    labels = {"description":"Description","model_load":"Model Load","embedding":"Embedding","retrieval":"Retrieval","dedup":"Dedup"}
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


# page_text 읽기 (프롬프트 입력 X, 임베딩 블렌딩용 O)
def read_page_text_for_image(img_path: Path, folder: Path) -> str:
    """
    이미지 파일명에 포함된 'page{n}' 정보를 이용해 같은 폴더의 text_page{n}.txt를 읽어옴.
    임베딩 블렌딩(page_text)만을 위해 사용.
    """
    page = next((seg[4:] for seg in img_path.stem.split("_") if seg.startswith("page")), None)
    if not page: return ""
    txt_path = folder / f"text_page{page}.txt"
    if txt_path.exists():
        for enc in ("utf-8","cp949","euc-kr"):
            try:
                return txt_path.read_text(encoding=enc).strip()
            except: pass
    return ""


# GPT 생성
def _build_messages(image_b64: str):
    prompt = PROMPT_TEMPLATE  # page_text는 프롬프트에 넣지 않음
    return [{"role": "user", "content": [
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_b64}"}}
    ]}]


def _encode_image_to_b64(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB"); im.thumbnail((512,512), RESAMPLE_LANCZOS)
        buf = BytesIO(); im.save(buf, format="JPEG", quality=90)
    return b64encode(buf.getvalue()).decode()


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


def describe_image(args):
    path = args
    b64 = _encode_image_to_b64(path)
    def _do():
        return _client().chat.completions.create(
            model="gpt-4o",  # "gpt-4.1-mini"
            temperature=TEMPERATURE,
            max_tokens=300,
            messages=_build_messages(b64),
        )
    try:
        rsp = _retry(_do)
        desc = rsp.choices[0].message.content.strip()
        usage = rsp.usage
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
    except Exception as e:
        print(f"[ERROR] OpenAI call failed for {path}: {e}")
        desc, tin, tout = "", 0, 0
    return str(path), desc, tin, tout

def generate_descriptions(folder: Path, tracker):
    imgs = sorted([p for p in folder.glob("*.png")])
    in_tok = out_tok = 0

    # 블렌딩용 page_text 미리 수집(키는 str 경로와 맞춤)
    page_text_map_all: Dict[str, str] = {str(p): read_page_text_for_image(p, folder) for p in imgs}

    raw_descs: Dict[str, str] = {}

    tracker.step_start()
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        for p, d, tin, tout in tqdm(
            ex.map(describe_image, imgs),
            total=len(imgs),
            desc="GPT 설명"
        ):
            raw_descs[p] = d
            in_tok += tin
            out_tok += tout
    tracker.step_end("description")

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

    descs: Dict[str, str] = {}
    page_texts: Dict[str, str] = {}  # <- 선택된(to_use=True) 이미지에 대해서만 유지
    all_image_paths: List[str] = []
    all_descriptions: List[str] = []
    all_to_use: List[bool] = []

    skipped_count = 0
    bad_samples = []

    for p, raw in raw_descs.items():
        obj = _extract_json(raw)
        to_use = bool(obj.get("to_use", False))
        desc = obj.get("description")
        all_image_paths.append(p)
        all_descriptions.append(desc.strip() if isinstance(desc, str) else "")
        all_to_use.append(to_use)

        if to_use and isinstance(desc, str) and desc.strip():
            descs[p] = desc.strip()
            page_texts[p] = page_text_map_all.get(p, "")
        else:
            skipped_count += 1
            bad_samples.append(p)

    cost = in_tok * 2.5 / 1e6 + out_tok * 10 / 1e6
    avg = cost / max(len(raw_descs), 1)

    DESCRIPTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DESCRIPTION_OUT_DIR / f"{folder.name}.csv"
    pd.DataFrame({
        "image_path": all_image_paths,
        "description": all_descriptions,
        "to_use": all_to_use
    }).to_csv(out_path, index=False, encoding="utf-8-sig")

    if skipped_count:
        print(f"[INFO] {skipped_count} images skipped for retrieval (to_use=false or invalid JSON). Examples: {bad_samples[:5]}")

    return descs, page_texts, cost, avg


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


# 메인 파이프라인
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
            # 페이지 텍스트는 이미지별로 다시 로드 (블렌딩용)
            page_texts = {p: read_page_text_for_image(Path(p), folder) for p in descs.keys()}
            cost, avg = 0, 0
        else:
            descs, page_texts, cost, avg = generate_descriptions(folder, tracker)

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

        # 1) Description 임베딩
        img_keys = list(descs.keys())
        desc_list = [descs[k] for k in img_keys]
        desc_emb = model.encode(
            desc_list,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE_EMB,
            show_progress_bar=True
        )

        # 2) Page text 임베딩 (블렌딩용)
        if USE_BLEND and PAGE_WEIGHT > 0:
            pt_list = [page_texts.get(k, "") for k in img_keys]
            text_emb = model.encode(
                pt_list,
                normalize_embeddings=True,
                batch_size=BATCH_SIZE_EMB,
                show_progress_bar=True
            )
            blended = DESC_WEIGHT * desc_emb + PAGE_WEIGHT * text_emb
            gallery_emb = l2norm(blended)
            tag = f"blend_{DESC_WEIGHT:.2f}:{PAGE_WEIGHT:.2f}"
        else:
            gallery_emb = desc_emb
            tag = "descOnly"

        # 쿼리 필터링
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
        img_paths_order = img_keys  # 삽입 순서 유지
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