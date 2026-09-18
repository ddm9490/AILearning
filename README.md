# bypp — AILearning

관심 분야를 입력하면 AI(Gemini)가 검색 키워드를 뽑아 arXiv에서 논문을 찾고, 그중에서
다시 AI가 내 관심사에 맞는 논문을 골라 핵심 기술/아키텍처/수학 키워드와 읽기 전 필요한
선수 지식을 함께 보여주는 웹앱입니다. 논문 하나든, "Transformer"/"CNN" 같은 순수
키워드든, 그걸 이해하기 위한 **커리큘럼(지식 노드 맵)**도 만들어줍니다 — 선형 목록이
아니라 독립된 선수 개념들이 가지를 이루며 목표로 수렴하는 진짜 그래프(DAG)입니다.
커리큘럼은 저장되고, 노드마다 학습 완료 표시·AI 추가 설명·이해도 확인 퀴즈까지 딸려
있어서 "논문을 찾아준다"에서 끝나지 않고 "실제로 배우게" 하는 걸 목표로 합니다.

## 주요 기능

- 관심 분야 자유 입력 + 추천 키워드 칩 다중 선택, 그리고 동그란 **+** 버튼으로 정확한
  키워드(예: "ConvNeXt")를 직접 추가할 수 있습니다 — 이렇게 추가한 키워드는 AI가
  다듬지 않고 그대로 검색에 씁니다 (AI에게 "정제"를 맡기면 정확한 고유명사도 다른
  말로 바꿔버리는 문제가 있어서, 정확히 아는 용어는 파이프라인에서 아예 LLM을 거치지
  않게 분리했습니다)
- Gemini API가 자연어 입력을 arXiv 검색에 적합한 나머지 키워드로 정제 → arXiv 실시간
  검색 (최대 5개 키워드까지 합쳐서 검색)
- Gemini API가 후보 논문 중 관심사에 맞는 논문을 골라 관련도 순으로 정렬하고, 논문마다
  추천 키워드(기술/아키텍처/필요 수학 지식)와 "읽기 전 필요한 선수 지식"을 붙여줌
- 논문 카드의 "커리큘럼 만들기", 또는 "커리큘럼" 탭의 "키워드로 커리큘럼 만들기"에서
  인터랙티브 지식 노드 맵(DAG) 생성 — 노드를 클릭하면 설명/학습 포인트/키워드가 보이고,
  호버하면 연결된 관계가 강조됩니다
- RAG는 켜고 끌 수 있습니다: 논문 커리큘럼은 실제 PDF를, 키워드 커리큘럼은 관련 논문
  초록 + 오픈 라이선스 학습 자료 3종([Dive into Deep Learning](https://d2l.ai)(CC
  BY-SA 4.0), [Hugging Face NLP Course](https://huggingface.co/course)(Apache 2.0),
  [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com)(MIT)) 발췌를 함께
  근거로 쓰거나, 꺼서 AI의 사전 지식만으로 만들 수도 있습니다. 그라운딩 소스는
  provider 하나만 갈아끼우면 바뀌는 pluggable 구조로 되어 있습니다
- **커리큘럼은 저장됩니다** — "내 커리큘럼" 탭에서 예전에 만든 것도 다시 볼 수 있고,
  진행률(완료 노드 수 / 전체 노드 수)이 미니 진행률 바로 표시됩니다. 노드마다:
  - **학습 완료로 표시**: 누르면 즉시 체크 배지로 표시되고 취소도 가능
  - **AI에게 더 자세히 설명 요청**: 그 노드에 집중한 RAG를 새로 돌려 깊은 설명을 생성
    (한 번 생성하면 캐싱되어 다시 눌러도 API를 또 호출하지 않음)
  - **이해도 확인 퀴즈**: 4지선다 객관식 문제를 생성해서 실제로 이해했는지 확인 —
    채점은 서버 호출 없이 프론트에서 즉시 처리되고, 통과하면 그 자리에서 학습 완료로
    표시할 수 있음 (캐싱되어 재요청 시 API를 또 부르지 않음)
  - **다음 학습 추천**: 선수 노드가 전부 완료된, 지금 바로 공부해도 되는 노드를
    LLM 호출 없이 그래프 순회만으로 계산해서 점선 테두리 + 클릭 가능한 칩으로 보여줌
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
GEMINI_API_KEY=your-api-key-here
```

```bash
python3 app.py
```

브라우저에서 http://localhost:5000 접속 (5000번 포트가 macOS AirPlay 수신기와 겹치면
`app.py`의 포트를 바꿔서 실행하세요). 첫 실행 시 로컬 임베딩 모델(67MB)이 자동으로
다운로드됩니다. 커리큘럼/저장 데이터는 SQLite(`data/app.db`)에 자동으로 생성됩니다.

키워드 커리큘럼에 D2L/HF Course/Spinning Up 내용을 근거로 쓰려면 (`git` CLI 필요,
1회성, 10~15분 정도 소요 — 소스 하나만 다시 빌드하려면 `python3 textbook_index.py d2l`
처럼 이름을 붙여서 실행):

```bash
python3 textbook_index.py
```

이 단계를 건너뛰어도 나머지 기능은 다 동작합니다 — 교재 발췌 없이 arXiv 초록만으로
그라운딩합니다.

## 프로젝트 구조

```
app.py                  Flask 서버, 라우트
arxiv_service.py         arXiv 검색 연동
llm_service.py           Gemini API 연동 (생성 전용: 키워드 정제/논문 큐레이션/커리큘럼 DAG/노드 설명/퀴즈)
embedding_service.py     로컬 임베딩 연동 (fastembed, RAG 검색 전용)
pdf_service.py           논문 PDF 다운로드/텍스트 추출/청크 분할
context_providers.py     커리큘럼 그라운딩(RAG) pluggable 전략
curriculum_service.py    커리큘럼(DAG) 생성 오케스트레이션 + 사이클 검증
curriculum_store.py       커리큘럼 저장/조회/삭제 + 노드 완료 상태 + AI 설명/퀴즈 캐싱 (SQLite)
textbook_index.py         D2L/HF Course/Spinning Up 오프라인 인덱싱 (로컬 파일 캐시, 벡터 DB 없음)
keyword_catalog.py       키워드 추상화 단계(tier) 분류 체계
frontend/                정적 프론트엔드 (백엔드 API를 호출해 동적으로 렌더링, SVG DAG 뷰 포함)
```

## 주요 API

- `POST /api/recommend` — 관심사/키워드로 논문 추천
- `POST /api/curriculum` — 논문 또는 키워드를 목표로 커리큘럼(DAG) 생성 (자동 저장됨)
- `GET /api/curricula`, `GET /api/curriculum/<id>`, `DELETE /api/curriculum/<id>` — 저장된 커리큘럼 목록/조회/삭제
- `POST /api/curriculum/<id>/nodes/<node_id>/complete` — 노드 완료 토글
- `POST /api/curriculum/<id>/nodes/<node_id>/explain` — 노드 AI 추가 설명 (캐싱)
- `POST /api/curriculum/<id>/nodes/<node_id>/quiz` — 노드 이해도 확인 퀴즈 (캐싱, 클라이언트 채점)

자세한 요청/응답 형식과 설계 이유는 [CLAUDE.md](CLAUDE.md)를 참고하세요.

## 참고

- Gemini 무료 티어는 요청 수 제한이 있어서, 짧은 간격으로 여러 번 검색하면 요청
  하나가 최대 1분 가까이 걸릴 수 있습니다 (자동 재시도, 필요하면 보조 모델로 전환).
  임베딩은 로컬에서 돌아가서 이 제한과 무관합니다.
- 커리큘럼 생성은 PDF 다운로드/임베딩/Gemini 생성까지 거치기 때문에 첫 요청에 시간이
  좀 걸릴 수 있습니다. 대신 한 번 만든 커리큘럼은 자동으로 저장되고, "내 커리큘럼"
  탭에서 다시 열면 그대로 보여줍니다 — 새로고침해도 사라지지 않아요.
- 더 자세한 아키텍처/설계 이유는 [CLAUDE.md](CLAUDE.md)를 참고하세요.
