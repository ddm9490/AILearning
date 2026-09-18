"""Gemini API 연동.

관심사 -> 검색 키워드 정제, 후보 논문 -> 추천 논문 선별/키워드 태깅 두 단계를 맡는다.
Gemini 모델이 503(UNAVAILABLE)을 자주 반환하는 걸 직접 확인했으므로, 두 단계 모두
재시도 로직을 거친다.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.6-flash"
NUM_SEARCH_KEYWORDS = 5
RETRY_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 2
DEFAULT_RATE_LIMIT_DELAY = 20

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_KEY 환경변수가 설정되어 있지 않아요 (.env 확인).")
        _client = genai.Client(api_key=api_key)
    return _client


def _rate_limit_delay(message):
    # Gemini 무료 티어는 분당 요청 수가 낮아서(확인해보니 gemini-3.6-flash 기준 20RPM)
    # 429가 꽤 자주 난다. 메시지에 들어있는 "retry in Xs" 힌트를 최우선으로 쓴다.
    match = re.search(r"retry in ([\d.]+)s", message)
    if match:
        return float(match.group(1)) + 1
    return DEFAULT_RATE_LIMIT_DELAY


def _generate_json(prompt, schema):
    client = _get_client()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )

    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            return json.loads(response.text)
        except Exception as exc:  # Gemini가 503/429를 자주 반환해서 재시도가 꼭 필요하다.
            last_error = exc
            if attempt >= RETRY_ATTEMPTS - 1:
                break
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                delay = _rate_limit_delay(message)
            else:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)
            time.sleep(delay)
    raise last_error


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


def curate_papers(interest_text, selected_keywords, candidates, count):
    """후보 논문 중 관심사에 맞는 것을 고르고, 논문별 추천 키워드를 붙인다.

    candidates: [{"id", "title", "summary"}]
    반환: [{"id", "keywords": [str, ...]}] (관련도 높은 순)
    """
    combined = interest_text.strip()
    if selected_keywords:
        combined = f"{combined}\n선택한 키워드: {', '.join(selected_keywords)}".strip()

    listing = "\n\n".join(
        f"[{c['id']}] {c['title']}\n{c['summary'][:600]}" for c in candidates
    )

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
                    },
                    "required": ["id", "keywords"],
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

사용자 관심사와 가장 관련 있는 논문을 최대 {count}개 골라서, 관련도가 높은 순서로 나열해줘.
각 논문마다 이 논문에서 다루는 핵심 기술/아키텍처/필요한 수학 개념을 3~5개 키워드로 붙여줘.
키워드는 짧고 구체적으로: 기술/아키텍처 용어는 영어로(예: Self-Attention, Diffusion Model),
수학 개념은 한국어로(예: 선형대수, 베이즈 통계) 표기해줘. id는 후보 목록에 있는 것만 정확히 사용해."""

    result = _generate_json(prompt, schema)
    papers = result.get("papers", [])
    if not papers:
        raise RuntimeError("Gemini가 추천 논문을 반환하지 않았어요.")
    return papers[:count]
