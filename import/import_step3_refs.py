import re
from neo4j import GraphDatabase

# ✅ Neo4j Desktop 값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"

BATCH = 200  # Step2보다 좀 크게 해도 괜찮음

def openalex_work_id_from_url(url: str) -> str | None:
    # "https://openalex.org/W123" -> "W123"
    if not url:
        return None
    m = re.search(r"/(W\d+)$", str(url).strip())
    return m.group(1) if m else None

def ensure_index(tx):
    # Paper lookup을 빠르게 하기 위해 openalexId 인덱스/제약을 보장
    tx.run("CREATE CONSTRAINT paper_openalexId IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalexId IS UNIQUE")

def connect_cites_batch(tx, rows):
    """
    rows: [{"src": <paper_openalexId_url>, "refs": [<ref_url_or_id>, ...]}, ...]
    """
    cypher = """
    UNWIND $rows AS row
      MATCH (p:Paper {openalexId: row.src})
      UNWIND row.refs AS ref
        WITH p, ref
        MATCH (q:Paper)
        WHERE q.openalexId = ref
        MERGE (p)-[:CITES]->(q)
    """
    tx.run(cypher, rows=rows)

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver:
        with driver.session() as session:
            session.execute_write(ensure_index)

        total_src = 0
        total_refs_seen = 0
        total_batches = 0

        with driver.session() as session:
            # DB에서 referencedWorks를 직접 읽어서 연결 (파일 다시 안 읽어도 됨)
            skip = 0
            while True:
                result = session.run(
                    """
                    MATCH (p:Paper)
                    WHERE p.referencedWorks IS NOT NULL AND size(p.referencedWorks) > 0
                    RETURN p.openalexId AS src, p.referencedWorks AS refs
                    SKIP $skip LIMIT $limit
                    """,
                    skip=skip, limit=BATCH
                ).data()

                if not result:
                    break

                rows = []
                for r in result:
                    src = r["src"]
                    refs_raw = r.get("refs") or []

                    # refs가 URL이거나 W123 형태일 수 있으니 둘 다 정규화
                    refs_norm = []
                    for x in refs_raw:
                        if not x:
                            continue
                        s = str(x).strip()
                        # URL이면 그대로, W123이면 URL로 변환해 통일
                        if s.startswith("http"):
                            refs_norm.append(s)
                        else:
                            wid = openalex_work_id_from_url(s) or (s if s.startswith("W") else None)
                            if wid:
                                refs_norm.append(f"https://openalex.org/{wid}")

                    if not refs_norm:
                        continue

                    rows.append({"src": src, "refs": refs_norm})
                    total_src += 1
                    total_refs_seen += len(refs_norm)

                if rows:
                    def work(tx):
                        connect_cites_batch(tx, rows)
                    session.execute_write(work)
                    total_batches += 1

                print(f"[batch] sources={total_src}, refs_seen={total_refs_seen}, batches={total_batches}")
                skip += BATCH

        print("DONE.")
        print("Total source papers processed:", total_src)
        print("Total referencedWorks entries seen:", total_refs_seen)

if __name__ == "__main__":
    main()
