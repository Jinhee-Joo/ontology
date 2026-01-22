import json
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"

JSONL_PATH = "paper_annotations_all_v3.jsonl"

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with driver:
        with driver.session() as session:
            def work(tx):
                for row in rows:
                    pid = row.get("id")
                    refs = row.get("referenced_works", []) or []
                    # 너무 크면 저장 부담이니 상한 걸고 싶으면 [:200] 같은 슬라이스도 가능
                    tx.run(
                        """
                        MATCH (p:Paper {openalexId:$pid})
                        SET p.referencedWorks = $refs
                        """,
                        pid=pid, refs=refs
                    )
            session.execute_write(work)

    print("DONE: referencedWorks saved to Paper nodes")

if __name__ == "__main__":
    main()
