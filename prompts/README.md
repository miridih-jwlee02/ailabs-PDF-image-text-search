# Prompts Directory

AI 프레젠테이션 생성 파이프라인에서 사용되는 모든 프롬프트 템플릿을 관리하는 디렉토리입니다.


## 📁 디렉토리 구조

```
prompts/
├── format/                     # 각 단계별 포맷 프롬프트
│   └── v17-test/               # v17 테스트 버전 프롬프트
│       ├── outline-background-search-query-test/    # Level 1: 개요 생성용 웹서치 쿼리
│       ├── outline-topic-test/                      # Level 2: 프레젠테이션 개요 생성
│       ├── template-recommendation-test/            # Level 2.2: 템플릿 추천
│       ├── layout-test/                            # Level 3: 레이아웃 설계
│       ├── content-background-search-query-test/   # Level 4: 콘텐츠 생성용 웹서치 쿼리
│       ├── content-topic-test/                     # Level 5: 슬라이드 콘텐츠 생성
│       ├── content-instruction-test/               # Level 5: 콘텐츠 생성 지시사항
│       └── zip/                                    # 압축된 프롬프트 파일들
│
└── item/                       # 템플릿별 아이템 프롬프트
    ├── layout693988-prompt-v10.zip                 # 템플릿 693988용 프롬프트
    ├── layout693981-prompt-v10.zip                 # 템플릿 693981용 프롬프트
    ├── layout693972-prompt-v10.zip                 # 템플릿 693972용 프롬프트
    ├── layout693985-prompt-v10.zip                 # 템플릿 693985용 프롬프트
    ├── layout693975-prompt-v10.zip                 # 템플릿 693975용 프롬프트
    └── layout691848-prompt-v10.zip                 # 템플릿 691848용 프롬프트
```


## 포맷(format) 프롬프트

- AI의 역할과 행동 지침 정의
- 출력 형식 및 제약사항 명시
- 품질 기준 및 검증 규칙 포함
- 동적 데이터 삽입을 위한 템플릿
- `%s` 플레이스홀더를 통한 변수 치환
- 실제 사용자 입력 및 이전 단계 결과 포함

포맷 프롬프트의 각 단계별 디렉토리는 다음과 같은 구조를 가집니다:

```
[단계명]-test/
├── system-prompt.txt          # 시스템 프롬프트 (AI 역할 및 지시사항)
└── user-prompt.txt           # 사용자 프롬프트 템플릿 (변수 포함)
```

## 아이템(item) 프롬프트

각 레이아웃 템플릿에 특화된 프롬프트 패키지.
