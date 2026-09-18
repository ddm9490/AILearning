# bypp — AI 학습 사이트

관심 분야를 입력하면 AI(Gemini)가 검색 키워드를 뽑아 arXiv에서 논문을 찾고, 그중에서
다시 AI가 내 관심사에 맞는 논문을 골라 핵심 기술/아키텍처/수학 키워드와 읽기 전 필요한
선수 지식을 함께 보여주는 웹앱입니다. 논문 하나든, "Transformer"/"CNN" 같은 순수
키워드든, 그걸 이해하기 위한 **커리큘럼(지식 노드 맵)**도 만들어줍니다 — 선형 목록이
아니라 독립된 선수 개념들이 가지를 이루며 목표로 수렴하는 진짜 그래프(DAG)입니다.

## 주요 기능

- 관심 분야 자유 입력 + 추천 키워드 칩 다중 선택
- Gemini API가 입력을 arXiv 검색에 적합한 키워드 5개로 정제 → arXiv 실시간 검색
- Gemini API가 후보 논문 중 관심사에 맞는 논문을 골라 관련도 순으로 정렬하고, 논문마다
  추천 키워드(기술/아키텍처/필요 수학 지식)와 "읽기 전 필요한 선수 지식"을 붙여줌
- 논문 카드의 "커리큘럼 만들기", 또는 홈 화면의 "키워드로 커리큘럼 만들기" 패널에서
  인터랙티브 지식 노드 맵(DAG) 생성 — 노드를 클릭하면 설명/키워드가 보이고, 호버하면
  연결된 관계가 강조됩니다
- RAG는 켜고 끌 수 있습니다: 논문 커리큘럼은 실제 PDF를, 키워드 커리큘럼은 관련 논문
  초록 + [Dive into Deep Learning](https://d2l.ai)(CC BY-SA 4.0) 교재 발췌를 함께
  근거로 쓰거나, 꺼서 AI의 사전 지식만으로 만들 수도 있습니다
- 키워드는 추상화 단계(수학 기초 → 연산/메커니즘 → 레이어/학습 기법 → 최적화 기법 →
  소규모 아키텍처 → 대규모 아키텍처 → 상위 패러다임)에 따라 색으로 구분해서 표시

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` 파일에 Gemini API 키를 넣어야 합니다:

```
GEMINI_KEY=your-api-key-here
```

```bash
python3 app.py
```

브라우저에서 http://localhost:5000 접속 (5000번 포트가 macOS AirPlay 수신기와 겹치면
`app.py`의 포트를 바꿔서 실행하세요). 첫 실행 시 로컬 임베딩 모델(67MB)이 자동으로
다운로드됩니다.

키워드 커리큘럼에 D2L 교재 내용을 근거로 쓰려면 (`git` CLI 필요, 1회성, 5분 정도 소요):

```bash
python3 textbook_index.py
```

이 단계를 건너뛰어도 나머지 기능은 다 동작합니다 — 교재 발췌 없이 arXiv 초록만으로
그라운딩합니다.

## 프로젝트 구조

```
app.py                  Flask 서버, 라우트
arxiv_service.py         arXiv 검색 연동
llm_service.py           Gemini API 연동 (생성 전용)
embedding_service.py     로컬 임베딩 연동 (fastembed, RAG 검색 전용)
pdf_service.py           논문 PDF 다운로드/텍스트 추출/청크 분할
context_providers.py     커리큘럼 그라운딩(RAG) pluggable 전략
curriculum_service.py    커리큘럼(DAG) 생성 오케스트레이션
textbook_index.py         D2L 교재 오프라인 인덱싱 (로컬 파일 캐시, 벡터 DB 없음)
keyword_catalog.py       키워드 추상화 단계(tier) 분류 체계
frontend/                정적 프론트엔드 (백엔드 API를 호출해 동적으로 렌더링, SVG DAG 뷰 포함)
```

## 참고

- Gemini 무료 티어는 요청 수 제한이 있어서, 짧은 간격으로 여러 번 검색하면 요청
  하나가 최대 1분 가까이 걸릴 수 있습니다 (자동 재시도, 필요하면 보조 모델로 전환).
  임베딩은 로컬에서 돌아가서 이 제한과 무관합니다.
- 커리큘럼 생성은 PDF 다운로드/임베딩/Gemini 생성까지 거치기 때문에 첫 요청에 시간이
  좀 걸릴 수 있습니다. 한 번 만든 커리큘럼은 페이지를 새로고침하기 전까지 다시 요청하지
  않고 그대로 보여줍니다.
- 더 자세한 아키텍처/설계 이유는 [CLAUDE.md](CLAUDE.md)를 참고하세요.
