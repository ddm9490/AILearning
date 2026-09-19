"""유명 AI/ML 논문 몇 편의 예시 커리큘럼을 만들어, 모든 사용자에게 공유되는 읽기
전용 콘텐츠로 저장한다(curriculum_store.EXAMPLE_OWNER_ID) — 그리고 그 결과를
example_curricula.py에 정적 데이터로 그대로 써준다.

textbook_index.py와 같은 성격의 오프라인/1회성 스크립트 — 배포 서버가 매 요청마다
도는 경로가 아니다. 실제 Gemini/arXiv API를 호출하므로 quota를 소모하고 논문당
대체로 10~40초 걸린다. 여기서 만든 example_curricula.py를 커밋해두면, 앱은 그
파일을 시작할 때 그대로 심기만 하고(curriculum_store.seed_example_curricula_if_missing)
Gemini를 다시 호출하지 않는다 — 그래서 이 스크립트를 배포 파이프라인(Dockerfile
등)에 넣지 않고 로컬에서 필요할 때만 돌린다.

새 예시를 추가/갱신하고 싶으면 EXAMPLE_PAPERS를 고치고 다시 실행한 뒤,
git diff로 example_curricula.py 변경을 확인하고 커밋하면 된다.

실행: python3 seed_examples.py
"""

import time
from pprint import pformat

import arxiv

import arxiv_service
import curriculum_service
import curriculum_store

# arxiv_id를 직접 못박아둔다 — 처음엔 제목으로만 검색했는데("all:" 필드 관련도
# 검색), "Generative Adversarial Networks"처럼 흔한 단어 조합은 원 논문이 아닌
# 엉뚱한 논문(다른 논문의 초록/카테고리에 그 문구가 더 많이 등장하는 경우)이
# 1위로 잡히는 걸 실제로 겪었다. 예시로 보여줄 논문은 정확성이 중요해서, 제목
# 검색 대신 이미 정확히 알고 있는 arXiv id로 직접 조회한다.
EXAMPLE_PAPERS = [
    {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
    {"title": "Deep Residual Learning for Image Recognition", "arxiv_id": "1512.03385"},
    {"title": "Generative Adversarial Networks", "arxiv_id": "1406.2661"},
]

MODULE_HEADER = '''"""유명 AI/ML 논문들의 예시 커리큘럼 — 실제로 Gemini/arXiv 파이프라인을 돌려서
얻은 결과를 그대로 박아둔 정적 데이터.

앱이 시작될 때 DB에 예시가 하나도 없으면 curriculum_store.seed_example_curricula_if_missing()이
이 데이터를 그대로 심는다. 배포 환경(Fly.io)은 data/app.db용 영구 볼륨이 없어서
재배포마다 DB가 초기화되는데, 그때마다 Gemini/arXiv를 다시 호출해 예시를
재생성하면(예: Dockerfile 빌드 단계에 넣는 방식) 커밋할 때마다 quota를 쓰게 되고
API 키를 빌드 레이어에 노출해야 하는 문제가 있었다 — 그래서 "한 번 실제로
생성한 결과를 커밋해두고, 매 시작 시 그대로 집어넣기만" 하는 방식을 택했다.
새 예시를 추가/갱신하려면 seed_examples.py를 다시 돌리면(생성 + 이 파일 재작성을
한 번에 함) 된다. 이 파일은 손으로 고치지 말 것 — seed_examples.py가 매번
새로 써서 덮어쓴다.
"""

EXAMPLE_CURRICULA = '''

_client = arxiv.Client()


def _fetch_by_id(arxiv_id):
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    return results[0] if results else None


def _write_example_curricula_module(results):
    with open("example_curricula.py", "w") as f:
        f.write(MODULE_HEADER)
        f.write(pformat(results, width=100))
        f.write("\n")


def main():
    curriculum_store.init_db()

    # 로컬 DB에 이전에 심어둔 예시가 있으면 지우고 새로 만든다 — 스크립트를
    # 다시 돌릴 때마다 중복 저장되는 걸 막기 위함(운영 DB는 안 건드림, 이건
    # 로컬에서 example_curricula.py를 만들기 위한 작업용 DB일 뿐).
    for item in curriculum_store.list_example_curricula():
        curriculum_store.delete_curriculum(item["id"], curriculum_store.EXAMPLE_OWNER_ID)

    results = []
    for entry in EXAMPLE_PAPERS:
        title = entry["title"]
        print(f"조회 중: {title} (arXiv:{entry['arxiv_id']})")
        result_paper = _fetch_by_id(entry["arxiv_id"])
        if not result_paper:
            print(f"  건너뜀 — arXiv id를 찾지 못했어요: {entry['arxiv_id']}")
            continue

        paper = arxiv_service.to_dict(result_paper)

        start = time.time()
        try:
            result = curriculum_service.generate_curriculum(
                target_label=paper["title"],
                target_description=paper["summary"],
                target={"pdf_url": paper["pdf_url"], "title": paper["title"]},
                provider_name=curriculum_service.provider_name_for("paper", True, "ai_ml"),
                interest="",
                domain="ai_ml",
            )
        except Exception as exc:  # 하나 실패해도 나머지 논문은 계속 진행한다.
            print(f"  실패 — {title}: {exc}")
            continue

        result["target_type"] = "paper"
        result["domain"] = "ai_ml"
        result["pdf_url"] = paper["pdf_url"]
        result["target_description"] = paper["summary"]
        for node in result["nodes"]:
            node["completed"] = False
            node["ai_explanation"] = None
            node["quiz"] = None

        curriculum_store.save_curriculum(
            paper["title"], "paper", "ai_ml", result, curriculum_store.EXAMPLE_OWNER_ID
        )
        elapsed = time.time() - start
        print(f"  저장됨: {paper['title']} ({elapsed:.1f}초, 노드 {len(result['nodes'])}개)")
        results.append(result)

    if results:
        _write_example_curricula_module(results)
        print(f"\nexample_curricula.py에 {len(results)}개 예시를 썼어요. git diff로 확인 후 커밋해주세요.")
    else:
        print("\n생성된 예시가 하나도 없어서 example_curricula.py를 건드리지 않았어요.")


if __name__ == "__main__":
    main()
