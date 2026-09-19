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

from keyword_catalog import TIERS

load_dotenv()

# keywords/prerequisites/concepts를 만들 때 LLM에게 tier(0~6)도 같이 매기게 한다.
# keyword_catalog.resolve_keyword()가 카탈로그에 정확히 있는 용어는 항상 카탈로그
# 판정을 우선하지만(일관성 보장), 카탈로그에 없는 새 용어는 이 LLM 판정을 그대로
# 써서 회색(tier 없음) 태그가 되는 걸 막는다 — 사용자가 "어떤 키워드는 색이 있고
# 어떤 건 없어서 마음에 안 든다"고 리포트해서 추가함.
TIER_DEFINITIONS_TEXT = "\n".join(f"{t['id']}: {t['name']} — {t['description']}" for t in TIERS)

# TIERS(keyword_catalog.py)의 이름/설명은 "레이어 / 학습 기법", "신경망 레이어 구성
# 요소" 처럼 AI/ML 용어로 쓰여 있다 — tier id(0~6)와 프론트 색상 매핑은 분야와 무관하게
# 그대로 재사용하고 싶지만, 이 문구를 AI/ML이 아닌 분야 프롬프트에 그대로 주면 LLM이
# 생물학/물리학 개념을 억지로 "레이어"나 "신경망"에 비유해서 설명하려 든다. 그래서
# 같은 tier id 순서로, "얼마나 기초적인가(0) ↔ 얼마나 상위 개념인가(6)"라는 축의
# 의미만 분야 중립적으로 다시 쓴 설명을 별도로 둔다.
GENERIC_TIER_DESCRIPTIONS = [
    "그 분야를 이해하기 위한 전제가 되는 수학/이론적 기초",
    "구체적인 계산이나 절차 수준의 핵심 동작·메커니즘",
    "결과를 만들어내는 과정에서 쓰이는 세부 기법이나 구성 요소",
    "성능이나 결과를 개선하기 위해 조정하는 기법",
    "여러 요소가 묶여 하나의 기능 단위를 이루는 구성 블록·소규모 체계",
    "여러 구성 블록이 결합된 대표적인 이론·모델·시스템 전체 구조",
    "특정 이론·시스템을 넘어서는 상위 방법론이나 패러다임",
]
GENERIC_TIER_DEFINITIONS_TEXT = "\n".join(
    f"{t['id']}: {desc}" for t, desc in zip(TIERS, GENERIC_TIER_DESCRIPTIONS)
)

AI_ML_DOMAIN = "ai_ml"


def _tier_definitions_text(domain):
    return TIER_DEFINITIONS_TEXT if domain == AI_ML_DOMAIN else GENERIC_TIER_DEFINITIONS_TEXT


def _subject_phrase(domain):
    """프롬프트 페르소나 문장에 끼워 넣을 짧은 분야 표현. AI/ML 도메인은 실측으로
    검증된 기존 "AI/ML" 문구를 그대로 유지하고(회귀 방지), 그 외 분야는 "AI"를 특정
    분야로 안 좁히고 학습자의 관심사(interest_text) 자체에서 분야를 유추하게 둔다 —
    우리가 아직 분야별 전용 프롬프트를 준비하지 못했으니(물리학/생물학 등), 여기서
    섣불리 특정 분야 이름을 못박기보다 일반화된 표현을 쓰는 편이 더 넓게 맞는다."""
    return "AI/ML" if domain == AI_ML_DOMAIN else "이공계"

# quota가 모델별로 따로 관리되기 때문에, 주 모델이 일일 한도를 다 쓰면 순서대로
# 다음 모델로 넘어간다. gemini-3.1-flash-lite는 성능은 낮지만 별도 quota를 쓴다.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
RETRY_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 2
DEFAULT_RATE_LIMIT_DELAY = 20

_client = None


class QuotaExhaustedError(RuntimeError):
    """일/월 단위로 리셋되는 quota가 소진된 경우. 재시도로는 해결되지 않는다."""
 

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않아요 (.env 확인).")
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


def refine_search_keywords(interest_text, count, domain=AI_ML_DOMAIN):
    """자유 입력 관심사(자연어)만 arXiv 검색에 적합한 영어 키워드 count개로 정제한다.

    사용자가 직접 고르거나 입력한 "정확한" 키워드(예: "ConvNeXt")는 이 함수에 절대
    넣지 않는다 — LLM이 특정 아키텍처/모델 이름 같은 고유명사를 "이상하게 해석해서
    다른 키워드로 바꿔버리는" 문제를 실제로 겪어서(사용자 리포트), 정확한 키워드는
    app.py에서 이 함수를 거치지 않고 검색어에 그대로 합친다. 이 함수는 순수 자연어
    설명("~을 배우고 싶어요" 같은)을 검색어로 "번역"하는 역할만 한다.

    domain: "ai_ml"이면 실측으로 검증된 기존 AI/ML 전용 예시·페르소나를 그대로 쓴다.
    그 외(예: "other")는 특정 분야 이름을 못박지 않고(아직 분야별 전용 프롬프트를
    준비 못했으므로), interest_text 자체에서 분야를 유추하도록 일반화한다.
    """
    if count <= 0:
        return []

    schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["keywords"],
    }

    if domain == AI_ML_DOMAIN:
        persona = "너는 AI 논문 검색을 도와주는 어시스턴트야."
        example_hint = """각 키워드는 arXiv 논문 제목/초록에 실제로 등장할 법한 구체적인 기술/방법/분야 용어여야 해
(예: "reinforcement learning", "diffusion model", "graph neural network").
너무 포괄적이거나(예: "AI", "deep learning") 너무 좁은 표현은 피해줘."""
    else:
        persona = "너는 학술 논문 검색을 도와주는 어시스턴트야."
        example_hint = """각 키워드는 arXiv 논문 제목/초록에 실제로 등장할 법한, 이 관심사가 속한 학문
분야의 구체적인 기술/방법/현상/이론 용어여야 해. 관심사 자체가 속한 학문 분야를 먼저
파악한 뒤 그 분야 용어로 만들어줘. 너무 포괄적인 분야명(예: "physics", "biology") 자체를
키워드로 쓰거나 너무 좁아서 검색 결과가 거의 안 나올 표현은 피해줘."""

    prompt = f"""{persona} 아래는 사용자가 입력한 관심 분야야.

---
{interest_text.strip()}
---

이 관심사를 arXiv 검색에 적합한 영어 키워드/짧은 구문 정확히 {count}개로 바꿔줘.
{example_hint}"""

    result = _generate_json(prompt, schema)
    keywords = result.get("keywords", [])[:count]
    if not keywords:
        raise RuntimeError("Gemini가 검색 키워드를 반환하지 않았어요.")
    return keywords


def curate_papers(interest_text, selected_keywords, candidates, count, known_keywords=None, domain=AI_ML_DOMAIN):
    """후보 논문 중 관심사에 맞는 것을 고르고, 논문별 추천 키워드와 선수 지식을 붙인다.

    candidates: [{"id", "title", "summary"}]
    known_keywords: [{"name", "tier", "tier_name"}] — RAG로 후보 논문 텍스트에서
        미리 뽑아낸, keyword_catalog에 정의된 용어들. LLM이 이 목록을 참고해서
        표기를 통일하고, 카탈로그에 있는 신조어(SwiGLU, Flash Attention 등)를
        놓치지 않도록 근거로 준다.
    반환: [{"id", "reason": str, "keywords": [{name, tier}, ...], "prerequisites": [{name, tier}, ...]}]
    (관련도 높은 순). tier(0~6)는 LLM이 직접 매긴 판정이고, keyword_catalog.resolve_keyword()가
    카탈로그에 정확히 있는 용어는 이 값을 무시하고 카탈로그 판정을 우선 쓴다 — 카탈로그에
    없는 새 용어에 대해서만 이 tier가 실제로 쓰인다. reason은 "왜 이 논문을 골랐는지"를
    사용자 관심사에 구체적으로 연결한 1~2문장 — "왜 추천됐는지 알려달라"는 사용자
    피드백으로 추가됨.
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
                        "reason": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tier": {"type": "integer"},
                                },
                                "required": ["name", "tier"],
                            },
                            "minItems": 3,
                            "maxItems": 5,
                        },
                        "prerequisites": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tier": {"type": "integer"},
                                },
                                "required": ["name", "tier"],
                            },
                            "minItems": 2,
                            "maxItems": 4,
                        },
                    },
                    "required": ["id", "reason", "keywords", "prerequisites"],
                },
            }
        },
        "required": ["papers"],
    }

    persona = "너는 AI 논문 추천 어시스턴트야." if domain == AI_ML_DOMAIN else "너는 학술 논문 추천 어시스턴트야."
    prerequisite_example = (
        "선형대수, 확률론, 어텐션 메커니즘" if domain == AI_ML_DOMAIN else "이 관심사가 속한 분야의 기초 이론"
    )
    catalog_hint = (
        "특히 SwiGLU, Flash Attention처럼 최신이거나 헷갈리기 쉬운 용어는 네가 아는 표현 대신 이 목록의 표기를 우선해줘."
        if domain == AI_ML_DOMAIN
        else "특히 최신이거나 표기가 여러 갈래인 용어는 네가 아는 표현 대신 이 목록의 표기를 우선해줘."
    )
    keyword_naming_hint = (
        '기술/아키텍처/메커니즘 용어는 영어로(예: Self-Attention, Diffusion Model), 수학\n개념은 한국어로(예: 선형대수, 베이즈 통계) 표기해줘.'
        if domain == AI_ML_DOMAIN
        else "그 분야에서 통용되는 표기(전문 용어는 영어 원어를 유지해도 됨)를 그대로 써줘."
    )
    tier_definitions = _tier_definitions_text(domain)

    prompt = f"""{persona} 아래는 사용자의 관심사와, arXiv에서 찾은 후보 논문 목록이야.

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
써줘 — {catalog_hint} 물론 목록에 없어도 적절한 키워드/선수지식이 있으면 자유롭게
추가해도 돼):
---
{known_keywords_text}
---

사용자 관심사와 가장 관련 있는 논문을 최대 {count}개 골라서, 관련도가 높은 순서로 나열해줘.
사용자 관심사 문장에 단순 주제/키워드를 넘어서는 특별한 조건이나 선호가 있다면(예:
"최신 연구 위주로", "실무 적용 사례가 있는 논문이면 좋겠어요", "쉬운 논문으로",
"이론보다는 벤치마크 결과 중심으로") 논문 선택/순위에 그 조건을 실제로 반영하고,
reason에서 그 조건과 어떻게 연결되는지 언급해줘. 특별한 언급이 없으면 지금처럼
주제 관련도만으로 판단해줘.

각 논문마다 아래 세 가지를 붙여줘.

1. reason: 이 논문을 왜 골랐는지 사용자 관심사와 구체적으로 연결해서 1~2문장으로 설명해줘.
   "관련 있는 논문입니다" 같은 뭉뚱그린 말 말고, 관심사의 어떤 부분과 이 논문의 어떤
   내용(제목/초록에 실제로 있는 내용)이 어떻게 연결되는지 구체적으로 적어줘.
   **반드시 한국어 문장으로** 작성해줘(기술 용어는 원래 표기를 유지해도 됨).
2. keywords: 이 논문에서 다루는 핵심 기술/아키텍처/필요한 수학 개념 3~5개.
3. prerequisites: 이 논문을 읽기 "전에" 이미 알고 있어야 이해가 되는 선수 지식 2~4개
   (예: {prerequisite_example}). keywords가 "이 논문이 다루는 내용"이라면,
   prerequisites는 "이 논문을 읽는 독자가 미리 갖추고 있어야 할 배경 지식"이야.
   논문 난이도에 비해 사용자가 이미 잘 아는 개념(선택한 키워드에 있는 것)은 굳이
   반복하지 말고, 정말 필요한 배경 지식 위주로 골라줘.

keywords/prerequisites의 각 항목은 {{name, tier}} 객체야. name은 짧고 구체적으로:
{keyword_naming_hint} tier는 그 개념이 "얼마나
기초적인가(0) ↔ 얼마나 상위 개념인가(6)"를 아래 기준으로 0~6 중 하나로 매겨줘
(참고 지식 베이스에 있는 용어는 거기 표기를 우선하되, tier는 그 용어의 성격에
맞게 네가 직접 판단해서 매겨도 돼):
---
{tier_definitions}
---

id는 후보 목록에 있는 것만 정확히 사용해."""

    result = _generate_json(prompt, schema)
    papers = result.get("papers", [])
    if not papers:
        raise RuntimeError("Gemini가 추천 논문을 반환하지 않았어요.")
    return papers[:count]


def generate_curriculum(target_label, target_description, context_chunks, interest_text, known_keywords=None, domain=AI_ML_DOMAIN):
    """target_label(논문 제목이든 "Transformer" 같은 키워드든)을 이해하기 위한 선수
    개념들의 DAG(비순환 방향 그래프)를 만든다. 선형 목록이 아니라 진짜 그래프 —
    독립된 선수 개념은 서로 다른 가지(branch)로 존재할 수 있다.

    context_chunks: context_providers가 골라낸 근거 발췌문. 비어 있으면(RAG 없이)
    Gemini의 사전 지식만으로 만들라고 명시적으로 안내한다 — RAG on/off를 프롬프트
    레벨에서도 명확히 구분하는 것.
    domain: "ai_ml"이 아니면 페르소나/tier 정의 문구에서 AI/ML 용어를 빼서, LLM이
    다른 분야 개념을 억지로 신경망/딥러닝 용어에 비유하지 않게 한다.
    반환: {"nodes": [{"id","title","description","learning_points":[str,...],
           "concepts":[{name,tier},...],"is_target":bool}], "edges": [{"from","to"}]}
    (사이클 검증은 curriculum_service가 한다). concepts의 tier(0~6)는 LLM이 직접
    매긴 판정 — keyword_catalog.resolve_keyword()가 카탈로그에 없는 새 용어에
    한해서만 이 값을 대신 쓴다.
    """
    known_keywords = known_keywords or []
    known_keywords_text = ", ".join(kw["name"] for kw in known_keywords) if known_keywords else "(해당 없음)"

    if context_chunks:
        excerpts = "\n\n".join(f"[발췌 {i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
        context_section = f"""아래는 이 주제와 관련해 실제로 찾은 참고 자료 발췌문이야. 각 발췌 앞에는 출처가
대괄호로 표시되어 있어(예: [arXiv 논문 "..."], [D2L 교재], [이 논문 PDF 원문]).
description/learning_points를 쓸 때 이 발췌 내용에 최대한 구체적으로 근거하고,
"D2L 교재에서는 ~라고 설명한다", "OOO 논문에서는 ~를 제안했다"처럼 출처를 자연스럽게
녹여서 언급해줘 (일반론 말고 실제로 이 발췌에 쓰여 있는 내용으로):
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
                        "learning_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4,
                        },
                        "concepts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tier": {"type": "integer"},
                                },
                                "required": ["name", "tier"],
                            },
                            "minItems": 1,
                            "maxItems": 5,
                        },
                        "is_target": {"type": "boolean"},
                    },
                    "required": ["id", "title", "description", "learning_points", "concepts", "is_target"],
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

    coach_persona = (
        "너는 AI/ML 학습 커리큘럼을 설계하는 코치야." if domain == AI_ML_DOMAIN else "너는 학습 커리큘럼을 설계하는 코치야."
    )
    learning_point_example = (
        '"쿼리·키·값 벡터의 내적으로 어텐션 가중치를 계산하는 과정을 손으로 따라갈 수\n    있다"'
        if domain == AI_ML_DOMAIN
        else '"핵심 공식을 직접 유도하거나 대표 사례에 적용해볼 수 있다"'
    )
    concept_naming_hint = (
        "기술/아키텍처/메커니즘 용어는 영어로, 수학"
        if domain == AI_ML_DOMAIN
        else "그 분야에서 통용되는 표기(전문 용어는 영어 원어 유지 가능)로, 수학"
    )
    term_example = "Self-Attention, Backpropagation" if domain == AI_ML_DOMAIN else "그 분야의 전문 용어"
    tier_definitions = _tier_definitions_text(domain)

    # 논문/키워드는 누가 물어봐도 같은 걸 찾아주면 되지만, 커리큘럼은 "이 학습자"에게
    # 맞춰 만들어줄 수 있다는 게 이 서비스의 핵심 차별점이다 — interest_text가 그냥
    # 배경 정보로만 프롬프트에 꽂혀 있고 "이걸 실제로 어떻게 반영하라"는 지시가 없으면
    # LLM이 참고만 하고 넘어가기 쉽다. 그래서 interest_text가 있을 때만, 무엇을(이미
    # 아는 것 생략/모르는 것 확장, 응용·이론·최신동향 같은 방향성, 속도·깊이 선호)
    # 어떻게 반영해야 하는지 구체적인 축을 짚어서 지시한다.
    personalization_instruction = ""
    if interest_text.strip():
        personalization_instruction = """
학습자의 관심사/맥락 문장을 커리큘럼 설계에 **실제로 반영**해줘 — 그냥 배경 정보로
참고만 하고 넘어가지 마:
- 학습자가 이미 안다고 언급한 개념(예: "CNN은 이미 알아요")은 노드를 따로 만들지
  않거나 아주 간단히만 다뤄서 불필요한 반복을 줄여줘. 반대로 "처음이에요", "잘
  몰라요"라고 한 부분은 더 잘게 쪼개서 자세히 다뤄줘.
- "실무 적용 사례 위주로", "최신 연구 동향이 궁금해요", "수식보다는 직관 위주로"
  같은 방향성이 있으면 관련 노드를 추가하거나 description/learning_points에서 그
  관점을 강조해줘.
- "빠르게 핵심만", "꼼꼼하게 기초부터" 같은 속도·깊이 선호가 있으면 노드 개수와
  각 description의 상세도를 그에 맞게 조정해줘(단, 노드 개수는 아래 4~14개 범위를
  벗어나면 안 돼).
- 특별한 조건이 없는 문장이면 지금처럼 표준적인 기초→상위 개념 순서로 만들어줘.
"""

    prompt = f"""{coach_persona} 학습자의 최종 목표는
"{target_label}"을(를) 제대로 이해하는 거야.
{target_description_block}{interest_block}
{personalization_instruction}
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
- 각 노드는 다음을 가져야 해:
  - id(짧고 고유한 문자열, 예: "n1"), title(짧은 제목)
  - description: **3~6문장의 자세한 설명**. (1) 이 개념이 정확히 무엇인지,
    (2) 왜 이 순서에 있는지(선수 노드와 어떻게 연결되고, 다음 노드를 이해하는 데
    구체적으로 뭘 준비시켜주는지), (3) 참고 자료 발췌가 있으면 거기 있는 구체적인
    내용(수식, 용어, 논문 제목 등)을 실제로 언급. 뭉뚱그린 일반론(예: "이건 중요한
    개념입니다") 말고, 그 발췌에만 있는 디테일로 채워줘.
  - learning_points: 이 노드에서 **실제로 할 수 있어야 하는 것/이해해야 하는 것**을
    2~4개의 짧고 구체적인 항목으로. "OOO을 이해한다" 같은 뭉뚱그린 문장 말고
    {learning_point_example}처럼 행동 가능한(actionable) 문장으로 써줘.
  - concepts(이 노드와 관련된 핵심 키워드 1~5개, 각각 {{name, tier}} 객체)
- concepts의 name은 짧고 구체적으로: {concept_naming_hint}
  개념은 한국어로 표기해줘. tier는 그 개념이 "얼마나 기초적인가(0) ↔ 얼마나 상위
  개념인가(6)"를 아래 기준으로 0~6 중 하나로 매겨줘(참고 지식 베이스에 있는
  용어는 표기를 우선하되, tier는 성격에 맞게 네가 직접 판단해도 돼):
---
{tier_definitions}
---
- title/description/learning_points는 **반드시 한국어 문장으로** 작성해줘. 참고 자료
  발췌가 영어여도 그대로 옮기지 말고 한국어로 번역/설명해줘 — 그 안에 나오는 기술
  용어(예: {term_example})나 수식 자체는 원래 표기를 유지해도 되지만,
  설명 문장 자체가 영어여서는 안 돼."""

    result = _generate_json(prompt, schema)
    if not result.get("nodes"):
        raise RuntimeError("Gemini가 커리큘럼을 반환하지 않았어요.")
    return result


def explain_concept(target_label, node_title, node_description, context_chunks, known_keywords=None, domain=AI_ML_DOMAIN):
    """커리큘럼 노드 하나를 사용자가 "더 자세히 설명해줘"라고 요청했을 때 쓴다.
    커리큘럼 생성 때는 노드 하나당 3~6문장이 전부였는데, 여기서는 그 노드 하나에만
    집중해서 훨씬 깊게(여러 단락) 들어간다. 결과는 curriculum_store에 캐싱되므로
    같은 노드에 대해 이 함수가 두 번 불릴 일은 없다(버튼을 다시 눌러도 저장된 걸 보여줌).
    """
    known_keywords = known_keywords or []
    known_keywords_text = ", ".join(kw["name"] for kw in known_keywords) if known_keywords else "(해당 없음)"

    if context_chunks:
        excerpts = "\n\n".join(f"[발췌 {i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
        context_section = f"""아래는 이 개념과 관련해 실제로 찾은 참고 자료 발췌문이야. 설명에 최대한
구체적으로 반영해줘(출처가 태그로 표시되어 있으면 자연스럽게 인용해도 좋아):
---
{excerpts}
---"""
    else:
        context_section = "참고 자료 없이, 네가 알고 있는 지식만으로 작성해줘."

    schema = {
        "type": "object",
        "properties": {"explanation": {"type": "string"}},
        "required": ["explanation"],
    }

    tutor_persona = (
        "너는 AI/ML 개념을 깊이 있게 설명하는 튜터야." if domain == AI_ML_DOMAIN else "너는 개념을 깊이 있게 설명하는 튜터야."
    )
    term_example = "Self-Attention" if domain == AI_ML_DOMAIN else "그 분야의 전문 용어"
    prompt = f"""{tutor_persona} 학습자는 지금 "{target_label}"을
이해하기 위한 커리큘럼을 따라가는 중이고, 그중 "{node_title}"이라는 단계에서 "더 자세히
설명해줘"라고 요청했어.

지금까지 이 단계에 붙어있던 짧은 설명:
---
{node_description}
---

{context_section}

참고 지식 베이스 (해당되면 이 표기를 우선 써줘): {known_keywords_text}

위 짧은 설명보다 **훨씬 깊고 구체적인** 설명을 만들어줘 (4~8문장 또는 여러 짧은 단락):
- 이 개념을 실제로 이해했다고 할 수 있으려면 무엇을 알아야 하는지 구체적으로.
- 가능하면 구체적인 예시, 수식, 비유 중 하나 이상을 포함해줘.
- 흔히 헷갈리거나 오해하는 지점이 있다면 짚어줘.
- 참고 자료가 있다면 그 내용을 일반론이 아니라 실제로 인용/반영해줘.
- **반드시 한국어 문장으로만** 작성해줘. 참고 자료 발췌가 영어여도 그대로 옮기지 말고
  한국어로 번역/설명해줘 — 기술 용어(예: {term_example})나 수식 표기는 원래 형태를
  유지해도 되지만, 설명 문장 자체가 영어여서는 안 돼."""

    result = _generate_json(prompt, schema)
    explanation = result.get("explanation", "").strip()
    if not explanation:
        raise RuntimeError("Gemini가 설명을 반환하지 않았어요.")
    return explanation


def generate_quiz(target_label, node_title, node_description, context_chunks, known_keywords=None, domain=AI_ML_DOMAIN):
    """커리큘럼 노드 하나에 대한 이해도 확인 퀴즈(객관식 3~5문제)를 만든다.
    "학습 완료" 버튼이 자기 신고제라 실제 이해를 검증할 방법이 없다는 문제를 풀기 위한
    기능 — explain_concept과 같은 패턴으로 그 노드 제목에 집중한 RAG 근거를 쓰고,
    결과는 curriculum_store에 캐싱돼서 같은 노드에 대해 두 번 생성되지 않는다.
    """
    known_keywords = known_keywords or []
    known_keywords_text = ", ".join(kw["name"] for kw in known_keywords) if known_keywords else "(해당 없음)"

    if context_chunks:
        excerpts = "\n\n".join(f"[발췌 {i + 1}] {chunk}" for i, chunk in enumerate(context_chunks))
        context_section = f"""아래는 이 개념과 관련해 실제로 찾은 참고 자료 발췌문이야. 문제를 낼 때
가능하면 이 내용에 근거해서 구체적으로 만들어줘:
---
{excerpts}
---"""
    else:
        context_section = "참고 자료 없이, 네가 알고 있는 지식만으로 작성해줘."

    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "correct_index": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "options", "correct_index", "explanation"],
                },
                "minItems": 3,
                "maxItems": 5,
            }
        },
        "required": ["questions"],
    }

    quizmaster_persona = (
        "너는 AI/ML 개념의 이해도를 확인하는 퀴즈를 만드는 출제자야."
        if domain == AI_ML_DOMAIN
        else "너는 개념의 이해도를 확인하는 퀴즈를 만드는 출제자야."
    )
    term_example = "Self-Attention" if domain == AI_ML_DOMAIN else "그 분야의 전문 용어"
    prompt = f"""{quizmaster_persona} 학습자는 "{target_label}"을
이해하기 위한 커리큘럼을 따라가는 중이고, 그중 "{node_title}"이라는 단계를 "학습 완료"로
표시하기 전에 실제로 이해했는지 확인하고 싶어해.

이 단계에 붙어있는 설명:
---
{node_description}
---

{context_section}

참고 지식 베이스 (해당되면 문제/보기에 이 표기를 우선 써줘): {known_keywords_text}

"{node_title}"에 대한 4지선다 객관식 퀴즈를 3~5문제 만들어줘.
- 단순 암기(정의를 그대로 물어보는 것)보다는, 계산해보기/적용하기/비교하기처럼 실제로
  이해해야 풀 수 있는 문제 위주로 만들어줘.
- options는 정확히 4개, 그럴듯한 오답(흔히 헷갈리는 지점)을 포함해줘.
- correct_index는 0부터 시작하는 정답 인덱스(0~3).
- explanation은 정답인 이유(및 왜 다른 보기들이 틀렸는지)를 1~2문장으로.
- 참고 자료가 있다면 그 내용을 실제로 반영한 구체적인 문제를 만들어줘.
- question/options/explanation은 **반드시 한국어 문장으로만** 작성해줘. 참고 자료
  발췌가 영어여도 그대로 옮기지 말고 한국어로 번역해줘 — 기술 용어(예: {term_example})나
  수식 표기는 원래 형태를 유지해도 되지만, 문장 자체가 영어여서는 안 돼."""

    result = _generate_json(prompt, schema)
    questions = result.get("questions", [])
    # correct_index가 범위를 벗어나는 등 LLM이 스키마를 지켜도 내용이 이상할 수 있어서
    # 방어적으로 걸러낸다 — 잘못된 문제를 하나 버리는 게 퀴즈 전체를 실패시키는 것보다 낫다.
    valid_questions = [q for q in questions if 0 <= q.get("correct_index", -1) < len(q.get("options", []))]
    if not valid_questions:
        raise RuntimeError("Gemini가 유효한 퀴즈를 반환하지 않았어요.")
    return valid_questions
