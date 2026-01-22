import json
from neo4j import GraphDatabase

# ✅ Neo4j Desktop 기본값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"   

JSONL_PATH = "paper_annotations_all_v3.jsonl"

def upsert_paper(tx, openalex_id, title, year, tier):
    tx.run("""
    MERGE (p:Paper {openalexId: $openalex_id})
    SET p.title = $title,
        p.year = $year,
        p.tier = $tier
    """, openalex_id=openalex_id, title=title, year=year, tier=tier)

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    count = 0
    with driver.session() as session, open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            openalex_id = obj.get("id")
            if not openalex_id:
                continue

            title = obj.get("title", "")
            year = obj.get("year")
            tier = obj.get("tier")

            session.execute_write(upsert_paper, openalex_id, title, year, tier)
            count += 1

            # 진행상황 로그 (100개마다)
            if count % 100 == 0:
                print(f"Inserted/Updated: {count} papers")

    driver.close()
    print(f"DONE. Total papers inserted/updated: {count}")

if __name__ == "__main__":
    main()
