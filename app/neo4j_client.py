import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()  # 루트의 .env 로드

class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not password:
            raise RuntimeError("NEO4J_PASSWORD가 비어있어. .env를 확인해줘.")

        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, cypher: str, params: dict | None = None):
        params = params or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]
