import json
from neo4j import GraphDatabase

# ✅ Neo4j Desktop 값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"  # 너가 쓰던 비번 그대로

JSONL_PATH = "paper_annotations_all_v3.jsonl"

# (annotation_key, node_label, rel_type)
MAP = [
    ("ResearchTask", "ResearchTask", "HAS_RESEARCH_TASK"),
    ("Method",       "Method",       "USES_METHOD"),
    ("Software",     "Software",     "USES_SOFTWARE"),
    ("Taxon",        "Taxon",        "STUDIES_TAXON"),
    ("DataType",     "DataType",     "USES_DATATYPE"),
    ("Context",      "Context",      "HAS_CONTEXT"),
    ("Assumption",   "Assumption",   "ASSUMES"),
]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def upsert_annotations(tx, paper_id, ann_key, node_label, rel_type, items):
    # items: [{"label":..., "confidence":..., "evidence":...}, ...]
    # evidence 키가 evidenceText/evidence 로 섞일 수 있어 방어적으로 처리
    cypher = f"""
    MATCH (p:Paper {{openalexId: $paper_id}})
    UNWIND $items AS it
      WITH p, it,
           it.label AS label,
           coalesce(it.confidence, 0.0) AS conf,
           coalesce(it.evidence, it.evidenceText, "") AS ev
      MERGE (x:{node_label} {{label: label}})
      MERGE (p)-[r:{rel_type}]->(x)
      SET r.confidence = conf,
          r.evidenceText = ev
    """
    tx.run(cypher, paper_id=paper_id, items=items)

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    data = load_jsonl(JSONL_PATH)

    total_papers = 0
    total_rels = 0

    # 너무 크게 보내면 429처럼 “API”가 아니라 Neo4j 트랜잭션이 버거울 수 있어서 배치로
    BATCH = 50

    with driver:
        for batch in chunks(data, BATCH):
            with driver.session() as session:
                def work(tx):
                    nonlocal total_papers, total_rels
                    for row in batch:
                        paper_id = row.get("id") or row.get("openalexId")
                        if not paper_id:
                            continue

                        ann = row.get("_annotation", {})
                        # Paper가 없는 경우(혹시 step1 누락) 대비: Paper는 여기서 만들지 않고 skip
                        # (원하면 여기서 MERGE(p:Paper...) 추가도 가능)
                        for ann_key, node_label, rel_type in MAP:
                            items = ann.get(ann_key, []) or []
                            # 빈 리스트면 패스
                            if not items:
                                continue
                            # label 없는 것 제거
                            items = [it for it in items if isinstance(it, dict) and it.get("label")]
                            if not items:
                                continue

                            upsert_annotations(tx, paper_id, ann_key, node_label, rel_type, items)
                            total_rels += len(items)

                        total_papers += 1

                session.execute_write(work)
            print(f"[batch] processed papers: {total_papers}, rel-items attempted: {total_rels}")

    print("DONE.")
    print("Total papers scanned:", total_papers)
    print("Total rel-items attempted:", total_rels)

if __name__ == "__main__":
    main()
