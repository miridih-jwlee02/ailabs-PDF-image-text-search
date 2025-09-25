"""
한꺼번에 실행 (PDF별 tqdm 진행바 + 처리시간 로그 저장 + 경고/노이즈 분리)
"""
import time
import re
import csv
import sys
from pathlib import Path
import fitz
from tqdm import tqdm
from contextlib import redirect_stderr

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import warnings
warnings.filterwarnings("ignore", message="Parameter `strict_text` has been deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*is deprecated")


def get_pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count


def extract_numeric_id(stem: str) -> str:
    m_end = re.search(r'(\d+)$', stem)
    if m_end:
        return m_end.group(1)
    m_any = re.search(r'(\d+)', stem)
    if m_any:
        return m_any.group(1)
    return stem


def process_single_pdf(pdf_path: Path, converter: DocumentConverter, output_dir: Path, position: int, err_sink) -> dict:
    file_start = time.time()
    log = {
        "pdf_name": pdf_path.name,
        "output_name": "",
        "pages_total": 0,
        "pages_success": 0,
        "pages_failed": 0,
        "elapsed_sec": 0.0,
        "status": "ok",
        "error": ""
    }

    try:
        total_pages = get_pdf_page_count(pdf_path)
        log["pages_total"] = total_pages
    except Exception as e:
        log["status"] = "fail"
        log["error"] = f"page_count_error: {e}"
        log["elapsed_sec"] = time.time() - file_start
        return log

    numeric_id = extract_numeric_id(pdf_path.stem)
    output_path = output_dir / f"pdf_text_example_{numeric_id}.txt"
    log["output_name"] = output_path.name

    texts = []

    with tqdm(
        total=total_pages,
        desc=pdf_path.name,
        position=position,
        leave=True,
        dynamic_ncols=True,
        unit="page",
        file=sys.stdout,      # 진행바는 stdout으로 고정
    ) as pbar:
        for page_num in range(1, total_pages + 1):
            try:
                # 변환 구간에서만 stderr를 파일로 리다이렉트 → tqdm 줄 보호
                with redirect_stderr(err_sink):
                    conv_result = converter.convert(
                        source=str(pdf_path),
                        page_range=(page_num, page_num)
                    )
                    text = conv_result.document.export_to_text().strip()
                if text:
                    texts.append(text)
                    log["pages_success"] += 1
            except Exception:
                log["pages_failed"] += 1
            finally:
                pbar.update(1)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))
    except Exception as e:
        log["status"] = "fail"
        log["error"] = f"write_error: {e}"

    log["elapsed_sec"] = time.time() - file_start
    return log


def main():
    input_dir = Path("../data/pdf")
    output_dir = Path("../data/input_data")
    assert input_dir.exists(), f"❌ 입력 폴더가 없습니다: {input_dir}"

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = True # OCR 필요없으면 False, 속도 빨라짐.
    pipeline_opts.ocr_options = EasyOcrOptions(lang=["ko", "en"])
    pipeline_opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ PDF 파일을 찾지 못했습니다: {input_dir}")
        return

    print(f"🔎 총 {len(pdf_files)}개의 PDF를 처리합니다.\n")

    total_start = time.time()
    logs = []

    # 경고/노이즈 모을 파일 (stderr만 보냄)
    err_log_path = Path.cwd() / "noisy_warnings.log"
    with open(err_log_path, "a", buffering=1) as err_sink:
        for idx, pdf_path in enumerate(pdf_files):
            log = process_single_pdf(pdf_path, converter, output_dir, position=idx, err_sink=err_sink)
            logs.append(log)

    total_elapsed = time.time() - total_start

    # 처리 로그 CSV 저장
    log_filename = f"pdf_text_extraction_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_path = Path.cwd() / log_filename
    fieldnames = ["pdf_name", "output_name", "pages_total", "pages_success", "pages_failed", "elapsed_sec", "status", "error"]
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

    success_cnt = sum(1 for l in logs if l["status"] == "ok")
    fail_cnt = len(logs) - success_cnt
    print("\n===== 전체 처리 요약 =====")
    print(f"✅ 성공: {success_cnt}개 | ❌ 실패: {fail_cnt}개 | ⏱️ 총 소요 시간: {total_elapsed:.2f}초")
    print(f"📝 처리 로그: {log_path}")
    print(f"🧹 경고/노이즈 로그(stderr): {err_log_path}")


if __name__ == "__main__":
    main()