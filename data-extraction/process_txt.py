"""
배치 스크립트: PDF 텍스트(.txt) -> 이미지-텍스트 매핑 CSV 생성
- 해당 코드는 "miridih-dp-ai-presentation-feature-aippt-v17" 프로젝트의 코드를 기반으로 작성되었습니다.
- 입력:  ./data/input_data/*.txt
- 출력:  ./data/output_data/<stem>_image_text.csv
"""

import sys
from pathlib import Path
import json
import csv
import re
import traceback

INPUT_TXT_DIR = Path("../data/input_data")
OUTPUT_CSV_DIR = Path("../data/output_data")
PROMPTS_ROOT = Path("../prompts")
FORMAT_PROMPTS_PATH = PROMPTS_ROOT / "format" / "v17-test"

# 템플릿 아이템 프롬프트 zip (필요시 변경)
TEMPLATE_IDX = 691848
ITEM_PROMPT_ZIP = PROMPTS_ROOT / "item" / f"layout{TEMPLATE_IDX}-prompt-v10.zip"

from utils.prompt_io_utils import extract_formatprompts_from_folder
from utils.utils import read_text_file_auto_encoding
from utils.aippt_pipeline import (
    lv2_outline, lv2_2_template_recommendation, lv3_layout, lv5_content
)

# 설정값(사용자 입력 성격)
DEFAULT_AUDIENCE = "대학생"
DEFAULT_TONE = "전문적으로"
EXTRA_NOTE = "이 문서의 내용을 바탕으로 프레젠테이션을 만들어주세요. 한국어로 만들어주세요."
SELECTED_LANGUAGE = "ko"   # "ko" | "en" | "ja"
SEARCH_OUTPUT = "No Search Result"
SLIDE_COUNT_INPUT = "None"  # "None"이면 LLM이 자동 추정


# 해당 아이템 템플릿의 매핑규칙
IMAGE_TO_TEXT_MAP = {
    "1-0": ["1-4", "1-1"],
    "5-7": ["5-10", "5-15", "5-14"],
    "5-8": ["5-10", "5-13", "5-12"],
    "5-9": ["5-10", "5-17", "5-16"],
    "6-10": ["6-1", "6-8", "6-7"],
    "7-4": ["7-0", "7-7", "7-6"],
    "7-5": ["7-0", "7-9", "7-8"],
    "8-1": ["8-2", "8-8", "8-7"],
    "9-4": ["9-0"]
}


# JSON 파싱 유틸
#  - LLM 출력의 <json> 블록을 찾아 파싱을 최대한 시도
def extract_json_block(text: str) -> str:
    """문자열에서 JSON 블록만 추출"""
    if not isinstance(text, str):
        return ""
    if "<json>" in text:
        return text.split("<json>", 1)[1].strip()
    # fallback: 첫 '{' ~ 마지막 '}'
    l = text.find("{"); r = text.rfind("}")
    return text[l:r+1] if l != -1 and r != -1 and r > l else text

def _replace_smart_quotes(s: str) -> str:
    # “ ” ‘ ’ → " '
    return (s.replace("\u201C", '"')
             .replace("\u201D", '"')
             .replace("\u2018", "'")
             .replace("\u2019", "'"))

def clean_json_string(s: str) -> str:
    s = s.replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    s = _replace_smart_quotes(s)
    s = re.sub(r'\s+', ' ', s).strip()
    # 후행 쉼표 제거
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s

def fix_json_structure(json_str: str) -> str:
    """중괄호 균형 맞추기 등 간단한 구조 보정"""
    json_str = json_str.strip()
    open_br = json_str.count("{"); close_br = json_str.count("}")
    if open_br > close_br:
        json_str += "}" * (open_br - close_br)
    elif close_br > open_br:
        # 여분 제거 (단순 방식)
        excess = close_br - open_br
        for _ in range(excess):
            pos = json_str.rfind("}")
            if pos > 0:
                json_str = json_str[:pos] + json_str[pos+1:]
    return json_str

def parse_json_flexible(json_str: str) -> dict:
    """여러 방식으로 JSON 파싱 시도"""
    # 1) 기본
    try:
        return json.loads(json_str)
    except Exception:
        pass
    # 2) 정리 후
    try:
        return json.loads(clean_json_string(json_str))
    except Exception:
        pass
    # 3) 구조 보정 후
    try:
        s = fix_json_structure(clean_json_string(json_str))
        return json.loads(s)
    except Exception:
        pass
    # 4) 코드펜스 제거/간단 추출 재시도
    try:
        s = json_str.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            s = s[l:r+1]
        s = fix_json_structure(clean_json_string(s))
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}")


# 슬라이드 수 추출 (lv2_outline 출력에서)
def extract_slide_count(lv2_llm_output: str) -> int:
    try:
        blk = extract_json_block(lv2_llm_output)
        obj = json.loads(blk)
        return int(obj.get("slideCount", 0)) or 0
    except Exception:
        # fallback: 첫 숫자 탐색
        m = re.search(r'"slideCount"\s*:\s*(\d+)', lv2_llm_output)
        return int(m.group(1)) if m else 0


# lv5_content 결과 파싱 → CSV row 생성
def rows_from_lv5_content(lv5_data: dict) -> list:
    rows = []
    parsing_errors = 0

    for key, item in (lv5_data or {}).items():
        if not isinstance(item, dict) or "llm_output" not in item:
            continue

        json_block = extract_json_block(item["llm_output"])
        if not json_block:
            parsing_errors += 1
            continue

        try:
            parsed = parse_json_flexible(json_block)
        except Exception:
            parsing_errors += 1
            continue

        # 페이지 단위
        for page, content in parsed.items():
            if not isinstance(content, dict):
                continue
            text_items = content.get("TEXT", {})
            image_items = content.get("IMAGE", {})

            for image_number, text_numbers in IMAGE_TO_TEXT_MAP.items():
                if image_number not in image_items:
                    continue

                # 텍스트 묶기
                valid_texts = []
                for tn in text_numbers:
                    if tn in text_items:
                        t = text_items[tn]
                        if isinstance(t, str):
                            t = t.strip()
                            if t:
                                valid_texts.append(t)
                if not valid_texts:
                    continue

                image_content = image_items.get(image_number, "")
                if isinstance(image_content, str):
                    image_content = image_content.strip()
                else:
                    image_content = str(image_content)

                rows.append([
                    key,                      # lv5_key
                    page,                     # page id (ex: "1page")
                    ", ".join(text_numbers),  # text_number
                    "/".join(valid_texts),    # text
                    image_number,             # image_number
                    image_content             # image
                ])
    return rows


# 단일 .txt 파일 처리
def process_one_txt(txt_path: Path, format_prompts: dict) -> Path | None:
    try:
        # 1) 입력 텍스트 로드
        user_input_text = read_text_file_auto_encoding(str(txt_path))

        # 2) 파이프라인 실행
        lv2 = lv2_outline(
            format_prompts["outline-instruction-test"]["system-prompt"],
            format_prompts["outline-instruction-test"]["user-prompt"],
            user_input_text,
            SLIDE_COUNT_INPUT,
            DEFAULT_AUDIENCE,
            DEFAULT_TONE,
            EXTRA_NOTE,
            SEARCH_OUTPUT
        )

        lv2_2 = lv2_2_template_recommendation(
            format_prompts["template-recommendation-test"]["system-prompt"],
            format_prompts["template-recommendation-test"]["user-prompt"],
            user_input_text,
            DEFAULT_AUDIENCE,
            DEFAULT_TONE,
            str(PROJ_ROOT / "datasets" / "template_keywords.json"),
            SELECTED_LANGUAGE
        )

        # LLM이 추론한 slide 수 (안전 추출)
        slide_count = extract_slide_count(lv2["lv2_outline"]["llm_output"]) or "None"

        lv3 = lv3_layout(
            format_prompts["layout-test"]["system-prompt"],
            format_prompts["layout-test"]["user-prompt"],
            str(ITEM_PROMPT_ZIP),
            user_input_text,
            slide_count,
            DEFAULT_AUDIENCE,
            DEFAULT_TONE,
            EXTRA_NOTE,
            lv2["lv2_outline"]["llm_output"]
        )

        lv5 = lv5_content(
            format_prompts["content-instruction-test"]["system-prompt"],
            format_prompts["content-instruction-test"]["user-prompt"],
            str(ITEM_PROMPT_ZIP),
            user_input_text,
            slide_count,
            DEFAULT_AUDIENCE,
            DEFAULT_TONE,
            EXTRA_NOTE,
            lv2["lv2_outline"]["llm_output"],
            lv3["lv3_layout"]["llm_output"],
            SEARCH_OUTPUT
        )

        # 3) lv5_content 파싱 → rows
        lv5_data = lv5.get("lv5_content", {})
        rows = rows_from_lv5_content(lv5_data)

        # 4) CSV 저장
        OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUTPUT_CSV_DIR / f"{txt_path.stem}_image_text.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["lv5_key", "page", "text_number", "text", "image_number", "image"])
            w.writerows(rows)

        # 로그
        print(f"✅ {txt_path.name} → {out_csv.name} (rows={len(rows)})")
        return out_csv

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"❌ 실패: {txt_path.name} - {e}")
        traceback.print_exc()
        return None


# 메인
def main():
    # 프롬프트 로드
    if not FORMAT_PROMPTS_PATH.exists():
        raise FileNotFoundError(f"format prompts 경로가 없습니다: {FORMAT_PROMPTS_PATH}")
    format_prompts = extract_formatprompts_from_folder(str(FORMAT_PROMPTS_PATH))
    # 노이즈 키 제거 (노트북에서 zip 항목이 나오는 경우)
    if "zip" in format_prompts:
        del format_prompts["zip"]

    # 아이템 프롬프트 zip 체크
    if not ITEM_PROMPT_ZIP.exists():
        print(f"⚠️ 경고: item 프롬프트 zip이 없습니다: {ITEM_PROMPT_ZIP}")

    # 입력 .txt 수집
    if not INPUT_TXT_DIR.exists():
        raise FileNotFoundError(f"입력 디렉토리 없음: {INPUT_TXT_DIR}")
    txt_files = sorted(INPUT_TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ .txt 파일이 없습니다: {INPUT_TXT_DIR}")
        return

    print(f"🔎 총 {len(txt_files)}개 .txt 처리 시작\n")

    ok = 0
    for p in txt_files:
        out = process_one_txt(p, format_prompts)
        if out is not None:
            ok += 1

    print("\n===== 배치 처리 요약 =====")
    print(f"✅ 성공: {ok} / {len(txt_files)}")
    print(f"📁 출력 폴더: {OUTPUT_CSV_DIR}")

if __name__ == "__main__":
    main()
