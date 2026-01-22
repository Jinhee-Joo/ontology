# import_step2_annotations.py
import json
import re
from neo4j import GraphDatabase

# Neo4j Desktop 값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"

# scripts/import 폴더에서 실행 기준: v3 jsonl 위치
JSONL_PATH = "../data/raw/annotations/paper_annotations_all_v3.jsonl"

# (annotation_key, node_label, rel_type)
MAP = [
    ("ResearchTask",      "ResearchTask",      "HAS_RESEARCH_TASK"),
    ("Method",            "Method",            "USES_METHOD"),
    ("Software",          "Software",          "USES_SOFTWARE"),
    ("Taxon",             "Taxon",             "STUDIES_TAXON"),
    ("DataType",          "DataType",          "USES_DATATYPE"),
    ("GeoScope",          "GeoScope",          "HAS_GEOSCOPE"),
    ("EnvironmentType",   "EnvironmentType",   "HAS_ENVIRONMENT"),
    ("ContributionType",  "ContributionType",  "HAS_CONTRIBUTION"),
    # ("Context",         "Context",          "HAS_CONTEXT"),
    # ("Assumption",      "Assumption",       "ASSUMES"),
]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def openalex_work_id_from_url(url: str):
    # "https://openalex.org/W123" -> "W123"
    if not url:
        return None
    m = re.search(r"/(W\d+)$", url.strip())
    return m.group(1) if m else None

def normalize_items(raw):
    """
    v3 방어:
    - ["Biogeography", ...]  -> [{"name": "Biogeography"}...]
    - [{"label": "...", "confidence":..., "evidenceText":...}, ...] -> 표준화
    - null/빈문자열 제거
    """
    if not raw:
        return []

    out = []

    # 문자열 하나
    if isinstance(raw, str):
        raw = [raw]

    # 문자열 리스트
    if isinstance(raw, list) and raw and all(isinstance(x, str) or x is None for x in raw):
        for s in raw:
            if s and str(s).strip():
                out.append({"name": str(s).strip(), "confidence": None, "evidenceText": None})
        return out

    # dict 리스트 (v2/v3 스타일)
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("label")
            if not name or not str(name).strip():
                continue
            out.append({
                "name": str(name).strip(),
                "confidence": it.get("confidence"),
                "evidenceText": it.get("evidenceText") or it.get("evidence") or ""
            })
        return out

    # 그 외는 무시
    return []

def ensure_constraints(tx):
    # ✅ Paper는 openalexId(url)를 유니크로 (핵심)
    tx.run("CREATE CONSTRAINT paper_openalexId IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalexId IS UNIQUE")

    # 분류 노드 name 유니크
    labels = {node_label for _, node_label, _ in MAP}
    for lab in labels:
        tx.run(f"CREATE CONSTRAINT {lab.lower()}_name IF NOT EXISTS FOR (n:{lab}) REQUIRE n.name IS UNIQUE")

def upsert_annotations(tx, row, paper_id_url, node_label, rel_type, items):
    wid = openalex_work_id_from_url(paper_id_url)

    title = row.get("title", "") or ""
    year  = row.get("year", None)
    tier  = row.get("tier", None)

    cypher = f"""
    MERGE (p:Paper {{openalexId: $pid_url}})
      ON CREATE SET
        p.id = $pid_url,
        p.title = $title,
        p.year = $year,
        p.tier = $tier
      ON MATCH SET
        p.title = coalesce(p.title, $title),
        p.year  = coalesce(p.year,  $year),
        p.tier  = coalesce(p.tier,  $tier)

    SET p.openalexWorkId = coalesce(p.openalexWorkId, $wid)

    WITH p
    UNWIND $items AS it
      WITH p, it,
           it.name AS name,
           coalesce(it.confidence, 0.0) AS conf,
           coalesce(it.evidenceText, "") AS ev
      MERGE (x:{node_label} {{name: name}})
      MERGE (p)-[r:{rel_type}]->(x)
      SET r.confidence = conf,
          r.evidenceText = ev
    """
    tx.run(
        cypher,
        pid_url=paper_id_url,
        wid=wid,
        title=title,
        year=year,
        tier=tier,
        items=items
    )


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    data = load_jsonl(JSONL_PATH)

    total_rows_scanned = 0
    total_rel_items = 0
    BATCH = 50

    with driver:
        with driver.session() as session:
            session.execute_write(ensure_constraints)

        for batch in chunks(data, BATCH):
            with driver.session() as session:
                def work(tx):
                    nonlocal total_rows_scanned, total_rel_items
                    for row in batch:
                        paper_id = row.get("id") or row.get("openalexId")
                        if not paper_id:
                            continue

                        ann = row.get("_annotation", {}) or {}
                        if not isinstance(ann, dict):
                            continue

                        refs = row.get("referencedWorks") or row.get("referenced_works") or row.get("referencedWorks_ids") or []
                        if not isinstance(refs, list):
                            refs = []

                     
                        wid = openalex_work_id_from_url(paper_id)
                        tx.run(
                            """
                            MERGE (p:Paper {openalexId: $pid_url})
                            ON CREATE SET
                                p.id = $pid_url,
                                p.title = $title,
                                p.year = $year,
                                p.tier = $tier,
                                p.referencedWorks = $refs
                            ON MATCH SET
                                p.title = coalesce(p.title, $title),
                                p.year  = coalesce(p.year,  $year),
                                p.tier  = coalesce(p.tier,  $tier),
                                p.referencedWorks = coalesce(p.referencedWorks, $refs)

                            SET p.openalexWorkId = coalesce(p.openalexWorkId, $wid)
                            """,
                            pid_url=paper_id,
                            wid=wid,
                            title=row.get("title", "") or "",
                            year=row.get("year", None),
                            tier=row.get("tier", None),
                            refs=refs
                        )


                        # 그 다음 분류 관계들 upsert
                        for ann_key, node_label, rel_type in MAP:
                            raw = ann.get(ann_key)
                            items = normalize_items(raw)
                            if not items:
                                continue

                            upsert_annotations(tx, row, paper_id, node_label, rel_type, items)
                            total_rel_items += len(items)

                        total_rows_scanned += 1

                session.execute_write(work)

            print(f"[batch] rows_scanned={total_rows_scanned}, rel-items={total_rel_items}")

    print("DONE.")
    print("Total rows scanned:", total_rows_scanned)
    print("Total rel-items:", total_rel_items)

if __name__ == "__main__":
    main()
