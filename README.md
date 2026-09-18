# bypp — AI 논문 추천 웹앱

관심 분야를 입력하면 AI(Gemini)가 검색 키워드를 뽑아 arXiv에서 논문을 찾고, 그중에서
다시 AI가 내 관심사에 맞는 논문을 골라 핵심 기술/아키텍처/수학 키워드와 함께 보여주는
웹앱입니다.

## 주요 기능

- 관심 분야 자유 입력 + 추천 키워드 칩 다중 선택
- Gemini API가 입력을 arXiv 검색에 적합한 키워드 5개로 정제
- 정제된 키워드로 arXiv 실시간 검색
- Gemini API가 후보 논문 중 관심사에 맞는 논문을 골라 관련도 순으로 정렬하고, 논문마다
  추천 키워드(기술/아키텍처/필요 수학 지식)를 붙여줌
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
`app.py`의 포트를 바꿔서 실행하세요).

## 프로젝트 구조

```
app.py               Flask 서버, 라우트
arxiv_service.py      arXiv 검색 연동
llm_service.py        Gemini API 연동 (키워드 정제 + 논문 큐레이션)
keyword_catalog.py    키워드 추상화 단계(tier) 분류 체계
frontend/             정적 프론트엔드 (백엔드 API를 호출해 동적으로 렌더링)
```

## 참고

- Gemini 무료 티어는 분당 요청 수 제한이 있어서, 짧은 간격으로 여러 번 검색하면
  요청 하나가 최대 1분 가까이 걸릴 수 있습니다 (자동 재시도).
- 더 자세한 아키텍처/설계 이유는 [CLAUDE.md](CLAUDE.md)를 참고하세요.
