"""Gemini API 연동 (생성 전용).

관심사 -> 검색 키워드 정제, 후보 논문 -> 추천 논문 선별/키워드 태깅, 학습 로드맵
생성까지 세 가지 "생성" 작업을 맡는다. Gemini 모델이 503(UNAVAILABLE)을 자주
반환하는 걸 직접 확인했으므로 재시도 로직을 거친다. 또한 무료 티어의 일일 quota는
모델별로 따로 관리되므로(gemini-3.6-flash quota가 다 차도 gemini-3.1-flash-lite는
멀쩡함을 직접 확인함), 주 모델의 quota가 소진되면 보조 모델로 자동 전환한다.

임베딩(검색)은 이 파일이 아니라 embedding_service.py가 담당한다 — Gemini 임베딩은
무료 티어 quota가 금방 소진돼서(실측으로 확인함) 로컬 임베딩(fastembed)으로 옮겼다.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# quota가 모델별로 따로 관리되기 때문에, 주 모델이 일일 한도를 다 쓰면 순서대로
# 다음 모델로 넘어간다. gemini-3.1-flash-lite는 성능은 낮지만 별도 quota를 쓴다.
MODELS = ["gemini-3.1-flash-lite"]
# "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash", "gemini-3.5-flash-lite", 실제 배포떄 쓸 모델의 토큰을 아끼기 위해
NUM_SEARCH_KEYWORDS = 5
RETRY_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 2
DEFAULT_RATE_LIMIT_DELAY = 20

_client = None


class QuotaExhaustedError(RuntimeError):
    """일/월 단위로 리셋되는 quota가 소진된 경우. 재시도로는 해결되지 않는다."""
 

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_KEY 환경변수가 설정되어 있지 않아요 (.env 확인).")
        _client = genai.Client(api_key=api_key)
    return _client


def _rate_limit_delay(message):
    # 메시지에 들어있는 "retry in Xs" 힌트를 최우선으로 쓴다. 다만 이건 일/분 단위
    # quota를 구분하지 않고 항상 붙어있는 문구라서, 재시도해도 되는 경우에만 쓴다
    # (일 단위 quota 소진은 _is_daily_quota_exhausted 에서 먼저 걸러낸다).
    match = re.search(r"retry in ([\d.]+)s", message)
    if match:
        return float(match.group(1)) + 1
    return DEFAULT_RATE_LIMIT_DELAY


def _is_daily_quota_exhausted(message):
    # 실제로 겪어본 케이스: quotaId가 "...PerDay..."면 며칠 단위 quota가 다 찬 거라
    # 메시지의 "retry in 35s" 같은 힌트를 믿고 재시도해봤자 절대 성공하지 않는다.
    return "PerDay" in message


def _generate_with_model(client, config, model, prompt):
    """모델 하나에 대해서만 재시도한다. 일일 quota 소진이면 즉시(재시도 없이) 알린다."""
    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return json.loads(response.text)
        except Exception as exc:  # Gemini가 503/429를 자주 반환해서 재시도가 꼭 필요하다.
            message = str(exc)
            if _is_daily_quota_exhausted(message):
                raise QuotaExhaustedError(f"{model} 일일 quota 소진") from exc

            last_error = exc
            if attempt >= RETRY_ATTEMPTS - 1:
                break
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                delay = _rate_limit_delay(message)
            else:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)
            time.sleep(delay)
    raise last_error


def _generate_json(prompt, schema):
    client = _get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )

    for i, model in enumerate(MODELS):
        is_last_model = i == len(MODELS) - 1
        try:
            return _generate_with_model(client, config, model, prompt)
        except QuotaExhaustedError:
            if is_last_model:
                raise QuotaExhaustedError(
                    "Gemini 무료 티어의 일일 요청 한도를 다 썼어요 (보조 모델 포함). "
                    "내일 다시 시도하거나 유료 플랜으로 전환해주세요."
                )
            continue  # 다음 모델(보조 모델)로 넘어간다.


def refine_search_keywords(interest_text, selected_keywords):
    """자유 입력 관심사 + 선택된 키워드를 arXiv 검색에 적합한 영어 키워드 5개로 정제한다."""
    combined = interest_text.strip()
    if selected_keywords:
        combined = f"{combined}\n선택한 키워드: {', '.join(selected_keywords)}".strip()

    schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": NUM_SEARCH_KEYWORDS,
                "maxItems": NUM_SEARCH_KEYWORDS,
            }
        },
        "required": ["keywords"],
    }

    prompt = f"""너는 AI 논문 검색을 도와주는 어시스턴트야. 아래는 사용자가 입력한 관심 분야야.

---
{combined}
---

이 관심사를 arXiv 검색에 적합한 영어 키워드/짧은 구문 정확히 {NUM_SEARCH_KEYWORDS}개로 바꿔줘.
각 키워드는 arXiv 논문 제목/초록에 실제로 등장할 법한 구체적인 기술/방법/분야 용어여야 해
(예: "reinforcement learning", "diffusion model", "graph neural network").
너무 포괄적이거나(예: "AI", "deep learning") 너무 좁은 표현은 피해줘."""

    result = _generate_json(prompt, schema)
    keywords = result.get("keywords", [])[:NUM_SEARCH_KEYWORDS]
    if not keywords:
        raise RuntimeError("Gemini가 검색 키워드를 반환하지 않았어요.")
    return keywords


def curate_papers(interest_text, selected_keywords, candidates, count, known_keywords=None):
    """후보 논문 중 관심사에 맞는 것을 고르고, 논문별 추천 키워드와 선수 지식을 붙인다.

    candidates: [{"id", "title", "summary"}]
    known_keywords: [{"name", "tier", "tier_name"}] — RAG로 후보 논문 텍스트에서
        미리 뽑아낸, keyword_catalog에 정의된 용어들. LLM이 이 목록을 참고해서
        표기를 통일하고, 카탈로그에 있는 신조어(SwiGLU, Flash Attention 등)를
        놓치지 않도록 근거로 준다.
    반환: [{"id", "keywords": [str, ...], "prerequisites": [str, ...]}] (관련도 높은 순)
    """
    combined = interest_text.strip()
    if selected_keywords:
        combined = f"{combined}\n선택한 키워드: {', '.join(selected_keywords)}".strip()

    listing = "\n\n".join(
        f"[{c['id']}] {c['title']}\n{c['summary'][:600]}" for c in candidates
    )

    known_keywords = known_keywords or []
    if known_keywords:
        known_keywords_text = ", ".join(kw["name"] for kw in known_keywords)
    else:
        known_keywords_text = "(해당 없음)"

    schema = {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 5,
                        },
                        "prerequisites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4,
                        },
                    },
                    "required": ["id", "keywords", "prerequisites"],
                },
            }
        },
        "required": ["papers"],
    }

    prompt = f"""너는 AI 논문 추천 어시스턴트야. 아래는 사용자의 관심사와, arXiv에서 찾은 후보 논문 목록이야.

사용자 관심사:
---
{combined}
---

후보 논문 목록 (형식: [id] 제목 \\n 초록 일부):
---
{listing}
---

참고 지식 베이스 (이 후보 논문들 텍스트에서 실제로 발견된, 우리가 이미 정의해둔 정확한
용어들이야. keywords나 prerequisites를 고를 때 해당되는 게 있으면 아래 표기를 그대로
써줘 — 특히 SwiGLU, Flash Attention처럼 최신이거나 헷갈리기 쉬운 용어는 네가 아는 표현
대신 이 목록의 표기를 우선해줘. 물론 목록에 없어도 적절한 키워드/선수지식이 있으면
자유롭게 추가해도 돼):
---
{known_keywords_text}
---

사용자 관심사와 가장 관련 있는 논문을 최대 {count}개 골라서, 관련도가 높은 순서로 나열해줘.
각 논문마다 아래 두 가지를 붙여줘.

1. keywords: 이 논문에서 다루는 핵심 기술/아키텍처/필요한 수학 개념 3~5개.
2. prerequisites: 이 논문을 읽기 "전에" 이미 알고 있어야 이해가 되는 선수 지식 2~4개
   (예: 선형대수, 확률론, 어텐션 메커니즘). keywords가 "이 논문이 다루는 내용"이라면,
   prerequisites는 "이 논문을 읽는 독자가 미리 갖추고 있어야 할 배경 지식"이야.
   논문 난이도에 비해 사용자가 이미 잘 아는 개념(선택한 키워드에 있는 것)은 굳이
   반복하지 말고, 정말 필요한 배경 지식 위주로 골라줘.

키워드/선수지식은 짧고 구체적으로: 기술/아키텍처/메커니즘 용어는 영어로
(예: Self-Attention, Diffusion Model), 수학 개념은 한국어로(예: 선형대수, 베이즈 통계)
표기해줘. id는 후보 목록에 있는 것만 정확히 사용해."""

    result = _generate_json(prompt, schema)
    papers = result.get("papers", [])
    if not papers:
        raise RuntimeError("Gemini가 추천 논문을 반환하지 않았어요.")
    return papers[:count]


def generate_curriculum(target_label, target_description, context_chunks, interest_text, known_keywords=None):
    """target_label(논문 제목이든 "Transformer" 같은 키워드든)을 이해하기 위한 선수
    개념들의 DAG(비순환 방향 그래프)를 만든다. 선형 목록이 아니라 진짜 그래프 —
    독립된 선수 개념은 서로 다른 가지(branch)로 존재할 수 있다.

    context_chunks: context_providers가 골라낸 근거 발췌문. 비어 있으면(RAG 없이)
    Gemini의 사전 지식만으로 만들라고 명시적으로 안내한다 — RAG on/off를 프롬프트
    레벨에서도 명확히 구분하는 것.
    반환: {"nodes": [{"id","title","description","concepts":[str,...],"is_target":bool}],
           "edges": [{"from","to"}]} (사이클 검증은 curriculum_service가 한다)
    """
    known_keywords = known_keywords or []
    known_keywords_text = ", ".join(kw["name"] for kw in known_keywords) if known_keywords else "(해당 없음)"

    if context_chunks:
        excerpts = "\n\n".join(f"[발췌 {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks))
        context_section = f"""아래는 이 주제와 관련해 실제로 찾은 참고 자료 발췌문이야. description을 쓸 때
가능하면 여기 내용에 근거해서 구체적으로 써줘:
---
{excerpts}
---"""
    else:
        context_section = "참고 자료 없이, 네가 알고 있는 지식만으로 작성해줘."

    target_description_block = f"\n목표에 대한 설명: {target_description.strip()}\n" if target_description.strip() else ""
    interest_block = f"\n학습자의 관심사/맥락: {interest_text.strip()}\n" if interest_text.strip() else ""

    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 4,
                        },
                        "is_target": {"type": "boolean"},
                    },
                    "required": ["id", "title", "description", "concepts", "is_target"],
                },
                "minItems": 4,
                "maxItems": 14,
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                    },
                    "required": ["from", "to"],
                },
            },
        },
        "required": ["nodes", "edges"],
    }

    prompt = f"""너는 AI/ML 학습 커리큘럼을 설계하는 코치야. 학습자의 최종 목표는
"{target_label}"을(를) 제대로 이해하는 거야.
{target_description_block}{interest_block}
{context_section}

참고 지식 베이스 (아래 표기가 concepts에 해당되면 이 표기를 우선 써줘. 목록에 없어도
적절한 개념이 있으면 자유롭게 추가해도 돼):
---
{known_keywords_text}
---

"{target_label}"을 이해하기 위해 필요한 선수 개념들을 노드로, "먼저 알아야 하는 관계"를
간선(edge)으로 하는 학습 커리큘럼 DAG(비순환 방향 그래프)를 만들어줘.

- 노드는 4~14개. 가장 기초적인 개념부터 시작해서 "{target_label}" 자체를 나타내는
  노드로 수렴해야 해. "{target_label}"을 나타내는 노드는 정확히 하나만 만들고
  is_target을 true로 표시해 (나머지는 전부 false).
- 서로 관계없는 개념끼리 억지로 순서를 만들지 마. 독립적인 선수 개념은 별도의
  가지(branch)로 둬 — 일렬로 나열한 목록이 아니라 진짜 그래프 구조를 만드는 게 목표야
  (예: "선형대수"와 "확률/통계"는 서로 의존하지 않고 둘 다 다른 노드의 선수 조건일 수 있어).
- edges의 from/to는 반드시 nodes에 있는 id를 정확히 사용해. from은 "먼저 배워야 하는
  노드", to는 "그다음에 배우는 노드"야. 사이클(순환 참조)이 생기면 안 돼.
- 각 노드는 id(짧고 고유한 문자열, 예: "n1"), title(짧은 제목),
  description(1~2문장 설명 — 왜 필요한지, 컨텍스트가 있으면 구체적으로),
  concepts(이 노드에서 알아야 할 키워드 1~4개)를 가져야 해.
- concepts는 짧고 구체적으로: 기술/아키텍처/메커니즘 용어는 영어로, 수학 개념은
  한국어로 표기해줘."""

    result = _generate_json(prompt, schema)
    if not result.get("nodes"):
        raise RuntimeError("Gemini가 커리큘럼을 반환하지 않았어요.")
    return result
