"""
docling으로 페이지별로 페이지 내 (이미지, 텍스트) 동시 추출
"""
import os
import time
from pathlib import Path
import sys
import traceback

import psutil
from PyPDF2 import PdfReader
import torch
from PIL import Image
import numpy as np

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem


def process_pdf_single(input_pdf_path: str,
                        device: str = "cuda",
                        gpu_id: str = "0",
                        num_threads: int = 16) -> dict:
    input_pdf = Path(input_pdf_path)
    if not input_pdf.exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {input_pdf}")
        return {}

    chunk_name = f"{input_pdf.stem}_single"
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        os.environ["DOCLING_DEVICE"] = "cuda"
        print(f"▶ [{chunk_name}] 시작 on GPU {gpu_id}")
        print(f"🚀 CUDA ready: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["DOCLING_DEVICE"] = "cpu"
        print(f"▶ [{chunk_name}] 시작 on CPU")

    # Docling 파이프라인 옵션 설정
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=AcceleratorDevice.CUDA if use_cuda else AcceleratorDevice.CPU
    )
    pipeline_options.images_scale = 3.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_formula_enrichment = False
    pipeline_options.do_picture_classification = False
    pipeline_options.do_picture_description = False
    pipeline_options.force_backend_text = False

    try:
        from docling.datamodel.ocr_options import EasyOcrOptions
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=['ko', 'en'],
            force_full_page_ocr=True,
            bitmap_area_threshold=0.01,
            confidence_threshold=0.3,
            use_gpu=use_cuda
        )
    except:
        pass

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    settings.debug.profile_pipeline_timings = True

    proc = psutil.Process(os.getpid())
    cpu_samples, mem_samples = [], []

    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)
    print(f"📄 총 페이지 수: {total_pages}")

    output_dir = Path("./data/docling_parse_output") / chunk_name
    output_dir.mkdir(parents=True, exist_ok=True)

    success_pages = picture_count = table_count = skipped_pages = 0
    start_time = time.perf_counter()

    for page in range(1, total_pages + 1):
        try:
            cpu_samples.append(proc.cpu_percent(interval=None))
            mem_samples.append(proc.memory_info().rss / 1024 / 1024)

            conv_res = doc_converter.convert(str(input_pdf), page_range=(page, page))
            if not conv_res or not conv_res.document:
                skipped_pages += 1
                continue

            pictures = conv_res.document.pictures or []
            tables = conv_res.document.tables or []
            texts = conv_res.document.texts or []
            all_texts = [t.text.strip() for t in texts if hasattr(t, 'text') and t.text and t.text.strip()]

            img_items = list(pictures) + list(tables)
            if not img_items:
                success_pages += 1
                continue

            # 텍스트는 페이지당 하나의 파일로만 저장
            save_txt_path = output_dir / f"text_page{page:03d}.txt"
            if all_texts:
                with open(save_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_texts))

            for idx, element in enumerate(img_items):
                is_picture = isinstance(element, PictureItem)
                is_table = isinstance(element, TableItem)
                if not (is_picture or is_table):
                    continue

                if is_picture:
                    picture_count += 1
                    type_label = "picture"
                else:
                    table_count += 1
                    type_label = "table"

                filename = f"{type_label}_page{page:03d}_{idx:02d}.png"
                save_img_path = output_dir / filename

                try:
                    img = element.get_image(conv_res.document)
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                        if img.ndim == 3 and img.shape[0] in [1, 3]:
                            img = img.transpose(1, 2, 0)
                        if img.ndim == 3 and img.shape[2] == 1:
                            img = img.squeeze(2)
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    if isinstance(img, Image.Image):
                        img.save(save_img_path, "PNG")
                except:
                    continue

            success_pages += 1
        except:
            skipped_pages += 1
            continue

    elapsed = time.perf_counter() - start_time
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0.0

    print(f"\n✅ 처리 완료: {success_pages}/{total_pages} 페이지 성공")
    print(f"📊 평균 CPU: {avg_cpu:.2f}%, 평균 메모리: {avg_mem:.2f}MB")
    print(f"🖼️ 추출된 그림: {picture_count}, 📊 테이블: {table_count}, ⚠️ 실패: {skipped_pages}")
    print(f"⏱️ 전체 처리 시간: {elapsed:.2f}s")
    print(f"📁 출력 디렉토리: {output_dir}")

    return {
        "pdf": str(input_pdf),
        "success_pages": success_pages,
        "total_pages": total_pages,
        "elapsed_time_sec": elapsed,
        "avg_cpu": avg_cpu,
        "avg_mem": avg_mem,
        "picture_count": picture_count,
        "table_count": table_count,
        "skipped_pages": skipped_pages
    }


def main():
    print("🚀 PDF 이미지/테이블 + 페이지별 텍스트 추출 시작")

    input_dir = Path("./data/pdf")
    if not input_dir.exists():
        print(f"❌ 입력 폴더가 없습니다: {input_dir}")
        return

    # 🔎 폴더 내 모든 PDF 수집
    target_pdfs = sorted(input_dir.glob("*.pdf"))
    if not target_pdfs:
        print(f"⚠️ PDF 파일을 찾지 못했습니다: {input_dir}")
        return

    print(f"🔎 총 {len(target_pdfs)}개의 PDF를 처리합니다.\n")

    results = []
    for pdf_path in target_pdfs:
        try:
            print(f"\n📄 처리 대상: {pdf_path.name}")
            res = process_pdf_single(
                str(pdf_path),
                device="cuda" if torch.cuda.is_available() else "cpu",
                gpu_id="0",
                num_threads=16
            )
            if res:
                results.append(res)
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 중단되었습니다.")
            break
        except Exception as e:
            print(f"❌ 처리 중 오류({pdf_path.name}): {e}")
            continue

    # ✅ 최종 요약
    if results:
        print("\n===== 전체 처리 요약 =====")
        for r in results:
            name = Path(r["pdf"]).name
            print(f"- {name}: {r['success_pages']}/{r['total_pages']}p 성공, "
                    f"🖼 {r['picture_count']}개, 📊 {r['table_count']}개, "
                    f"⚠️ 실패 {r['skipped_pages']}p, ⏱ {r['elapsed_time_sec']:.2f}s")
    else:
        print("\n⚠️ 처리 결과가 없습니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print(traceback.format_exc())
        sys.exit(1)