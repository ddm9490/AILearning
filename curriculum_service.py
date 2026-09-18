"""커리큘럼(DAG) 생성 오케스트레이션.

app.py는 이 모듈의 generate_curriculum() 하나만 부르면 된다. 이 함수가:
1. target_type에 맞는 그라운딩 provider를 context_providers에서 골라 실행하고 (RAG on/off
   는 여기서 provider 하나 바꾸는 걸로 끝난다),
2. llm_service.generate_curriculum()으로 Gemini에게 DAG(nodes/edges)를 받고,
3. 그 DAG가 진짜 비순환인지 검증해서 사이클이 있으면 강제로 끊어내고,
4. 프론트가 별도 레이아웃 엔진 없이 바로 그릴 수 있도록 각 노드에 layer(선수 단계 깊이)를
   계산해 붙이고,
5. concepts를 keyword_catalog로 tier-resolve한다.
"""

from collections import defaultdict, deque

import context_providers
import llm_service
from keyword_catalog import extract_keywords, resolve_keyword

KNOWLEDGE_BASE_HITS_LIMIT = 30


def generate_curriculum(target_label, target_description, target, provider_name, interest):
    provider = context_providers.PROVIDERS.get(provider_name, context_providers.no_rag_provider)
    context_chunks = provider(target)

    # RAG로 얻은 발췌문이 있으면 거기서, 없으면 목표 자체(라벨+설명)에서 참고 지식
    # 베이스를 뽑는다 — 어느 쪽이든 흐름 1/2와 같은 사전 매칭 RAG 패턴을 재사용한다.
    grounding_text = " ".join(context_chunks) if context_chunks else f"{target_label} {target_description}"
    known_keywords = extract_keywords(grounding_text, limit=KNOWLEDGE_BASE_HITS_LIMIT)

    raw = llm_service.generate_curriculum(
        target_label=target_label,
        target_description=target_description,
        context_chunks=context_chunks,
        interest_text=interest,
        known_keywords=known_keywords,
    )

    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])

    layer_of, clean_edges = _validate_dag(nodes, edges)

    resolved_nodes = [
        {
            "id": node["id"],
            "title": node.get("title", ""),
            "description": node.get("description", ""),
            "learning_points": node.get("learning_points", []),
            # concepts는 이제 {name, tier} 객체 — tier는 LLM이 직접 매긴 판정으로,
            # resolve_keyword가 카탈로그/별칭 어디에서도 못 찾은 새 용어에 한해서만
            # 이 값을 대신 쓴다(카탈로그에 있으면 그쪽이 우선).
            "concepts": [
                resolve_keyword(c.get("name", ""), c.get("tier")) for c in node.get("concepts", [])
            ],
            "is_target": bool(node.get("is_target")),
            "layer": layer_of.get(node["id"], 0),
        }
        for node in nodes
    ]

    return {
        "target_label": target_label,
        "used_rag": bool(context_chunks),
        "nodes": resolved_nodes,
        "edges": clean_edges,
    }


def _validate_dag(nodes, edges):
    """Kahn's algorithm으로 위상 정렬 + layer 계산. LLM이 사이클을 만들어 보내면
    (스키마로는 막을 수 없다) 사이클을 완성하는 간선을 걸러내서 강제로 DAG로 만든다.
    """
    node_ids = {node["id"] for node in nodes}
    edges = [e for e in edges if e.get("from") in node_ids and e.get("to") in node_ids and e["from"] != e["to"]]

    children = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        children[edge["from"]].append(edge["to"])
        indegree[edge["to"]] += 1

    layer_of = {}
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    layer = 0
    while queue:
        next_queue = deque()
        for node_id in queue:
            layer_of[node_id] = layer
            for child in children[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_queue.append(child)
        queue = next_queue
        layer += 1

    if len(layer_of) < len(node_ids):
        # 사이클에 걸려 위상 정렬이 안 끝난 노드들: 마지막 레이어에 그냥 배치하고,
        # 그 노드로 "들어오는" 간선 중 layer 역행(사이클을 만드는) 간선만 제거한다.
        for node_id in node_ids:
            layer_of.setdefault(node_id, layer)
        edges = [e for e in edges if layer_of[e["from"]] < layer_of[e["to"]]]

    return layer_of, edges
