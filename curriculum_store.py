"""커리큘럼을 저장하고, 노드별 학습 완료 상태 + AI 추가 설명을 관리한다.

지금까지 커리큘럼은 매 요청마다 새로 만들고 응답이 끝나면 사라지는 일회성이었다.
이 모듈부터 "생성 -> 저장 -> 나중에 다시 보기 -> 노드별 진행 상황 추적"이 가능해진다.

SQLite를 쓴 이유: 표준 라이브러리라 새 의존성이 없고, 이 앱의 실제 규모(로컬에서
혼자 쓰는 개인용 앱, 커리큘럼 수십~수백 개)에 딱 맞는 만큼의 구조(관계형 테이블 +
쿼리, 여러 프로세스/스레드에서도 안전한 파일 기반 동시성)를 준다. 사용자 계정이
없는 로컬 앱이라 별도 인증/스코핑 없이 단일 저장소로 충분하다.

커리큘럼 본체(nodes/edges)는 한 번 생성되면 안 바뀌는 문서라서 JSON 그대로
저장한다(정규화된 테이블로 쪼갤 이유가 없음) — 진짜로 계속 바뀌는 부분(노드 완료
여부, AI 추가 설명)만 별도 테이블로 관리한다.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS curricula (
    id TEXT PRIMARY KEY,
    target_label TEXT NOT NULL,
    target_type TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL,
    owner_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS node_completions (
    curriculum_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    completed_at REAL NOT NULL,
    PRIMARY KEY (curriculum_id, node_id),
    FOREIGN KEY (curriculum_id) REFERENCES curricula(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS node_explanations (
    curriculum_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    explanation TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (curriculum_id, node_id),
    FOREIGN KEY (curriculum_id) REFERENCES curricula(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS node_quizzes (
    curriculum_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    quiz TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (curriculum_id, node_id),
    FOREIGN KEY (curriculum_id) REFERENCES curricula(id) ON DELETE CASCADE
);
"""


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        # 이미 만들어진 데이터베이스(예전 스키마)에도 owner_id를 추가한다 —
        # CREATE TABLE IF NOT EXISTS는 이미 있는 테이블의 컬럼을 바꿔주지 않아서
        # 별도로 ALTER TABLE이 필요하다. 기존 행(주인이 없던 커리큘럼)은 빈 문자열로
        # 채워지고, 그건 어떤 owner_id와도 매칭되지 않아 사실상 더 이상 아무도 못
        # 보게 된다 — 데모 전 로컬 테스트 데이터라 손실이어도 문제없음.
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(curricula)").fetchall()}
        if "owner_id" not in existing_cols:
            conn.execute("ALTER TABLE curricula ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def save_curriculum(target_label, target_type, result, owner_id):
    """generate_curriculum()이 반환한 {target_label, used_rag, nodes, edges}를 저장하고
    새로 만든 id를 돌려준다. owner_id는 브라우저별로 발급되는 익명 식별자(쿠키) —
    로그인이 없는 앱이라 "누가 만들었는지"를 구분하는 유일한 수단이다."""
    curriculum_id = uuid.uuid4().hex[:12]
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO curricula (id, target_label, target_type, created_at, payload, owner_id) VALUES (?, ?, ?, ?, ?, ?)",
            (curriculum_id, target_label, target_type, time.time(), json.dumps(result), owner_id),
        )
        conn.commit()
    finally:
        conn.close()
    return curriculum_id


def get_curriculum(curriculum_id, owner_id):
    """저장된 커리큘럼 + 노드별 완료 여부/AI 설명/퀴즈를 합쳐서 돌려준다. 없거나
    owner_id가 다르면(다른 사람 커리큘럼) None — 목록뿐 아니라 상세 조회/노드 조작도
    전부 이 함수를 거치므로 여기서 막으면 다른 사람 커리큘럼의 id를 알아내도 열람/조작이
    안 된다."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM curricula WHERE id = ? AND owner_id = ?", (curriculum_id, owner_id)
        ).fetchone()
        if not row:
            return None

        completed_ids = {
            r["node_id"]
            for r in conn.execute(
                "SELECT node_id FROM node_completions WHERE curriculum_id = ?", (curriculum_id,)
            ).fetchall()
        }
        explanation_map = {
            r["node_id"]: r["explanation"]
            for r in conn.execute(
                "SELECT node_id, explanation FROM node_explanations WHERE curriculum_id = ?", (curriculum_id,)
            ).fetchall()
        }
        quiz_map = {
            r["node_id"]: json.loads(r["quiz"])
            for r in conn.execute(
                "SELECT node_id, quiz FROM node_quizzes WHERE curriculum_id = ?", (curriculum_id,)
            ).fetchall()
        }
    finally:
        conn.close()

    payload = json.loads(row["payload"])
    for node in payload.get("nodes", []):
        node["completed"] = node["id"] in completed_ids
        node["ai_explanation"] = explanation_map.get(node["id"])
        node["quiz"] = quiz_map.get(node["id"])

    return {
        "id": row["id"],
        "target_type": row["target_type"],
        "created_at": row["created_at"],
        **payload,
    }


def list_curricula(owner_id):
    """이 owner_id가 만든 커리큘럼만, 생성 순 최신순으로, 목록 화면에 필요한 요약
    정보(진행률 등)만 돌려준다."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, target_label, target_type, created_at, payload FROM curricula "
            "WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
        summaries = []
        for row in rows:
            payload = json.loads(row["payload"])
            total = len(payload.get("nodes", []))
            completed = conn.execute(
                "SELECT COUNT(*) AS c FROM node_completions WHERE curriculum_id = ?", (row["id"],)
            ).fetchone()["c"]
            summaries.append(
                {
                    "id": row["id"],
                    "target_label": row["target_label"],
                    "target_type": row["target_type"],
                    "created_at": row["created_at"],
                    "total_nodes": total,
                    "completed_nodes": completed,
                }
            )
    finally:
        conn.close()
    return summaries


def set_node_completion(curriculum_id, node_id, completed):
    conn = _connect()
    try:
        if completed:
            conn.execute(
                "INSERT OR REPLACE INTO node_completions (curriculum_id, node_id, completed_at) VALUES (?, ?, ?)",
                (curriculum_id, node_id, time.time()),
            )
        else:
            conn.execute(
                "DELETE FROM node_completions WHERE curriculum_id = ? AND node_id = ?",
                (curriculum_id, node_id),
            )
        conn.commit()
    finally:
        conn.close()


def save_node_explanation(curriculum_id, node_id, explanation):
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO node_explanations
               (curriculum_id, node_id, explanation, created_at) VALUES (?, ?, ?, ?)""",
            (curriculum_id, node_id, explanation, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def save_node_quiz(curriculum_id, node_id, quiz):
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO node_quizzes (curriculum_id, node_id, quiz, created_at) VALUES (?, ?, ?, ?)",
            (curriculum_id, node_id, json.dumps(quiz), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def delete_curriculum(curriculum_id, owner_id):
    conn = _connect()
    try:
        conn.execute("DELETE FROM curricula WHERE id = ? AND owner_id = ?", (curriculum_id, owner_id))
        conn.commit()
    finally:
        conn.close()
