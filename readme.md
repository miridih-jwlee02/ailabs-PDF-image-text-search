# AIP PDF Image-Text Search

AI 프레젠테이션 생성을 위한 PDF 이미지-텍스트 매핑 및 검색 시스템입니다. PDF 문서에서 추출한 이미지와 텍스트를 분석하여 프레젠테이션 슬라이드에 적합한 이미지를 자동으로 매칭하는 기능을 제공합니다.

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 워크플로우를 제공합니다:

1. **PDF 처리**: PDF 문서에서 이미지와 텍스트를 추출
2. **이미지 분석**: GPT-4o를 사용하여 이미지에 대한 한국어 설명 생성
3. **텍스트 매핑**: 추출된 텍스트를 프레젠테이션 슬라이드 형태로 구조화
4. **유사도 검색**: 임베딩 기반으로 이미지와 텍스트 간 유사도 계산 및 매칭

## 🚀 주요 기능

### 1. PDF 데이터 추출

- **`extract_txt.py`**: PyMuPDF와 Docling을 사용한 PDF 텍스트 추출
- **`extract_image_txt_wt_docling.py`**: Docling을 활용한 이미지와 텍스트 동시 추출
- CUDA 가속 지원으로 빠른 처리 속도

### 2. AI 기반 이미지 분석

- **GPT 비전 모델**을 사용한 이미지 설명 생성
- 한국어 설명 자동 생성
- 프레젠테이션 적합성 판단 (`to_use` 플래그)
- 중복 이미지 자동 제거 (perceptual hashing)


### 3. 임베딩 기반 검색

- **Qwen3-Embedding-0.6B** 모델 사용
- 이미지 설명과 페이지 텍스트 블렌딩 (가중평균)
- 코사인 유사도 기반 Top-K 검색
- 성능 추적 및 시각화

## 🛠️ 설치 및 설정

### 필수 요구사항

```bash
# Python 패키지
pip install torch torchvision
pip install sentence-transformers
pip install openai anthropic
pip install docling docling-core
pip install PyMuPDF pandas numpy pillow
pip install imagehash tqdm wandb matplotlib
pip install python-dotenv chardet

# 시스템 요구사항
- Python 3.8+
- CUDA (GPU 가속 사용 시)
- Google Cloud CLI (웹서치 기능 사용 시)
```

### 환경 변수 설정

`.env` 파일을 생성하여 다음 환경 변수를 설정하세요:

```bash
# AWS Bedrock (Claude API)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token
BEDROCK_ANTHROPIC_MODEL_NAME=anthropic.claude-3-5-sonnet-20241022-v2:0

# OpenAI API
OPENAI_API_KEY=your_openai_api_key
```

## 📋 사용법

### 1. PDF에서 Raw 텍스트 추출

```bash
cd data-extraction
python extract_txt.py
```

### 2. 이미지와 텍스트 동시 추출 (Docling 사용)

```bash
cd data-extraction
python extract_image_txt_wt_docling.py
```

### 3. 텍스트를 프레젠테이션 구조로 변환

```bash
cd data-extraction
python process_txt.py
```

### 4. 이미지-텍스트 검색 실행

```bash
# 버전 6 사용
# 버전 6는 버전5와 동일한 모델이지만, 프롬프트에 (이미지 + 이미지가 들어있던 페이지의 텍스트)가 아닌, 이미지만 보고 Description을 생성. (비용 절감을 위한 실험)
python with_text_search_ver6.py

# 또는 버전 5 사용
# 최종 사용하기로 결정된 버전의 코드. (아래 주석처리한 버전의 코드를 이용하면 GPT-5버전도 실험 가능.)
python with_text_search_ver5.py
```

## ⚙️ 설정 옵션

### `with_text_search_ver6.py` 주요 설정

```python
# 모델 설정
TEMPERATURE = 0.2           # GPT 생성 온도
MAX_WORKERS = 10           # 병렬 처리 워커 수
BATCH_SIZE_EMB = 64        # 임베딩 배치 크기
TOP_K = 5                  # 검색 결과 상위 K개

# 임베딩 블렌딩 설정
USE_BLEND = True           # 블렌딩 사용 여부
DESC_WEIGHT = 0.6          # 이미지 설명 가중치
PAGE_WEIGHT = 0.4          # 페이지 텍스트 가중치

```

### `process_txt.py` 주요 설정

```python
# 기본 설정값
DEFAULT_AUDIENCE = "대학생"
DEFAULT_TONE = "전문적으로"
SELECTED_LANGUAGE = "ko"
TEMPLATE_IDX = 691848      # 사용할 템플릿 인덱스
```

## 📊 출력 결과

### 1. CSV 파일 (`output_data/`)
- `{filename}_image_text.csv`: 이미지-텍스트 매핑 결과
- 컬럼: 이미지 경로, 설명, 슬라이드 텍스트, 유사도 점수

### 2. 이미지 설명 파일 (`description/`)
- `{folder_name}.csv`: GPT가 생성한 이미지 설명
- 컬럼: `image_path`, `description`, `to_use`

### 3. 성능 차트
- `perf_{run}.png`: 각 단계별 처리 시간 분석
- WandB 로그: 검색 결과 시각화


## 📈 성능 최적화

- **GPU 가속**: CUDA 지원으로 임베딩 및 추론 가속화
- **배치 처리**: 이미지 설명 생성 및 임베딩 배치 처리
- **병렬 처리**: ThreadPoolExecutor를 사용한 동시 처리
- **메모리 최적화**: 이미지 크기 조정 및 압축
- **중복 제거**: perceptual hashing 기반 중복 이미지 제거

## 🔍 트러블슈팅

### 로그 및 디버깅
- 처리 시간은 자동으로 추적되어 차트로 저장됩니다
- WandB를 통해 검색 결과를 시각적으로 확인할 수 있습니다
- 각 단계별 토큰 사용량과 비용이 계산됩니다

## 🤝 기여

이 프로젝트는 "miridih-dp-ai-presentation-feature-aippt-v17" 프로젝트를 기반으로 개발되었습니다.