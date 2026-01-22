# app/streamlit_app.py
import os
import re
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pyvis.network import Network
import streamlit.components.v1 as components

# -----------------------------
# Config
# -----------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")  # 너 환경에 맞게 .env로 관리 권장

ONTO_REL_TYPES = [
    "USES_METHOD",
    "USES_DATATYPE",
    "USES_SOFTWARE",
    "HAS_CONTEXT",
    "HAS_RESEARCH_TASK",
    "STUDIES_TAXON",
    "ASSUMES",
]

ONTO_NODE_LABELS = ["Method", "DataType", "Software", "Context", "ResearchTask", "Taxon", "Assumption"]

# 관계 스타일(2번: edge 강조)
REL_STYLE = {
    "USES_METHOD": {"width": 2},
    "USES_DATATYPE": {"width": 2},
    "USES_SOFTWARE": {"width": 2},
    "HAS_CONTEXT": {"width": 2},
    "HAS_RESEARCH_TASK": {"width": 3},
    "STUDIES_TAXON": {"width": 3},
    "ASSUMES": {"width": 2},
    "CITES": {"width": 1, "dashes": True},
}

# 노드 스타일(라벨별)
NODE_STYLE = {
    "Paper": {"shape": "dot", "size": 28},
    "Method": {"shape": "dot", "size": 18},
    "DataType": {"shape": "dot", "size": 18},
    "Software": {"shape": "dot", "size": 18},
    "Context": {"shape": "dot", "size": 18},
    "ResearchTask": {"shape": "dot", "size": 20},
    "Taxon": {"shape": "dot", "size": 20},
    "Assumption": {"shape": "dot", "size": 18},
}

# -----------------------------
# Neo4j driver
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_cypher(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [r.data() for r in result]

# -----------------------------
# 3) Coverage / Metrics
# -----------------------------
def get_coverage_stats() -> Dict[str, Any]:
    q_total = "MATCH (p:Paper) RETURN count(p) AS total"
    total = run_cypher(q_total)[0]["total"]

    q_tagged = f"""
    MATCH (p:Paper)
    WHERE EXISTS {{
      MATCH (p)-[:{'|:'.join(ONTO_REL_TYPES)}]->()
    }}
    RETURN count(p) AS tagged
    """
    tagged = run_cypher(q_tagged)[0]["tagged"]

    # 관계별 커버리지: 해당 관계를 가진 Paper 수
    rel_rows = []
    for rel in ONTO_REL_TYPES:
        q = f"""
        MATCH (p:Paper)-[:{rel}]->()
        RETURN '{rel}' AS rel, count(DISTINCT p) AS papers
        """
        rel_rows.append(run_cypher(q)[0])

    rel_df = pd.DataFrame(rel_rows).sort_values("papers", ascending=False)

    return {
        "total": total,
        "tagged": tagged,
        "coverage": (tagged / total * 100.0) if total else 0.0,
        "rel_df": rel_df,
    }

# -----------------------------
# 1) Search with evidence
# -----------------------------
def search_papers_with_evidence(keyword: str, k: int) -> pd.DataFrame:
    """
    - title match + ontology tag match를 함께 돌리고
    - 결과에 reason(근거)를 붙임
    """
    kw = keyword.strip()
    if not kw:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])

    # 1) title match
    q_title = """
    MATCH (p:Paper)
    WHERE toLower(p.title) CONTAINS toLower($kw)
    RETURN p.openalexId AS openalexId, p.title AS title, p.year AS year,
           "title match" AS reason, 1.0 AS score
    LIMIT $k
    """

    # 2) ontology node(label/name/label property) match -> connected papers
    # NOTE: 노드 속성명이 name / label 둘 다 있을 수 있어서 둘 다 체크
    q_tag = f"""
    MATCH (t)
    WHERE any(lbl IN labels(t) WHERE lbl IN $onto_labels)
      AND (
        (exists(t.name)  AND toLower(t.name)  CONTAINS toLower($kw)) OR
        (exists(t.label) AND toLower(t.label) CONTAINS toLower($kw))
      )
    WITH t, labels(t)[0] AS tLabel,
         coalesce(t.name, t.label, "unknown") AS tName
    MATCH (p:Paper)-[r:{'|'.join(ONTO_REL_TYPES)}]->(t)
    RETURN p.openalexId AS openalexId, p.title AS title, p.year AS year,
           ("tag match: " + type(r) + " -> " + tLabel + " / " + tName) AS reason,
           2.0 AS score
    LIMIT $k
    """

    rows = []
    rows += run_cypher(q_title, {"kw": kw, "k": k})
    rows += run_cypher(q_tag, {"kw": kw, "k": k, "onto_labels": ONTO_NODE_LABELS})

    if not rows:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])

    df = pd.DataFrame(rows)

    # 같은 논문이 여러 reason으로 잡히면 score 합산 + reason은 합쳐서 보여줌
    agg = (
        df.groupby(["openalexId", "title", "year"], dropna=False)
          .agg(score=("score", "sum"), reason=("reason", lambda x: " | ".join(list(dict.fromkeys(x)))))
          .reset_index()
          .sort_values(["score", "year"], ascending=[False, False])
          .head(k)
    )
    return agg

# -----------------------------
# Paper detail (tags)
# -----------------------------
def get_paper_detail(openalex_id: str) -> Dict[str, Any]:
    q = f"""
    MATCH (p:Paper {{openalexId:$oid}})
    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
    OPTIONAL MATCH (p)-[:STUDIES_TAXON]->(t:Taxon)
    OPTIONAL MATCH (p)-[:USES_SOFTWARE]->(s:Software)
    OPTIONAL MATCH (p)-[:ASSUMES]->(a:Assumption)
    OPTIONAL MATCH (p)-[:HAS_CONTEXT]->(c:Context)
    OPTIONAL MATCH (p)-[:HAS_RESEARCH_TASK]->(rt:ResearchTask)
    OPTIONAL MATCH (p)-[:USES_DATATYPE]->(d:DataType)
    RETURN
      p.openalexId AS openalexId,
      p.title AS title,
      p.year AS year,
      collect(DISTINCT coalesce(m.name, m.label)) AS methods,
      collect(DISTINCT coalesce(t.name, t.label)) AS taxa,
      collect(DISTINCT coalesce(s.name, s.label)) AS software,
      collect(DISTINCT coalesce(a.name, a.label)) AS assumptions,
      collect(DISTINCT coalesce(c.name, c.label)) AS contexts,
      collect(DISTINCT coalesce(rt.name, rt.label)) AS researchTasks,
      collect(DISTINCT coalesce(d.name, d.label)) AS dataTypes
    """
    rows = run_cypher(q, {"oid": openalex_id})
    if not rows:
        return {
            "openalexId": openalex_id,
            "title": None,
            "year": None,
            "methods": [],
            "taxa": [],
            "software": [],
            "assumptions": [],
            "contexts": [],
            "researchTasks": [],
            "dataTypes": [],
        }
    detail = rows[0]

    # None 제거
    for key in ["methods", "taxa", "software", "assumptions", "contexts", "researchTasks", "dataTypes"]:
        detail[key] = [x for x in detail.get(key, []) if x not in (None, "")]
    return detail

# -----------------------------
# Similar papers + evidence
# -----------------------------
def get_similar_papers(openalex_id: str, k: int) -> pd.DataFrame:
    """
    공유 온톨로지 태그 기반 유사도.
    - 어떤 태그가 겹쳤는지(evidence) 같이 리턴
    """
    q = f"""
    MATCH (p:Paper {{openalexId:$oid}})
    // p가 가진 태그 노드들
    MATCH (p)-[r:{'|'.join(ONTO_REL_TYPES)}]->(t)
    WITH p, collect(DISTINCT t) AS tags

    // 다른 논문 p2가 tags 중 일부를 공유
    MATCH (p2:Paper)
    WHERE p2 <> p

    OPTIONAL MATCH (p2)-[r2:{'|'.join(ONTO_REL_TYPES)}]->(t2)
    WHERE t2 IN tags

    WITH p2,
         collect(DISTINCT {rel:type(r2), label:labels(t2)[0], name:coalesce(t2.name,t2.label)}) AS shared,
         count(DISTINCT t2) AS score
    WHERE score > 0
    RETURN p2.openalexId AS openalexId, p2.title AS title, p2.year AS year,
           score AS score,
           shared AS evidence
    ORDER BY score DESC, year DESC
    LIMIT $k
    """
    rows = run_cypher(q, {"oid": openalex_id, "k": k})
    if not rows:
        return pd.DataFrame(columns=["openalexId", "title", "year", "score", "evidence"])
    df = pd.DataFrame(rows)
    return df

# -----------------------------
# 2) Graph 1-hop with styled edges
# -----------------------------
def get_graph_1hop(openalex_id: str, max_edges: int) -> Tuple[List[Dict], List[Dict]]:
    """
    1-hop: (Paper)-[rel]->(tag) + (Paper)-[:CITES]->(Paper) 일부 포함 가능
    """
    q = f"""
    MATCH (p:Paper {{openalexId:$oid}})
    OPTIONAL MATCH (p)-[r:{'|'.join(ONTO_REL_TYPES)}]->(t)
    WITH p, collect({{from:id(p), to:id(t), type:type(r)}}) AS rels, collect(DISTINCT t) AS nodes
    RETURN p AS p, rels AS rels, nodes AS nodes
    """
    rows = run_cypher(q, {"oid": openalex_id})
    if not rows:
        return [], []

    p = rows[0]["p"]
    rels = [x for x in rows[0]["rels"] if x.get("to") is not None]
    nodes = rows[0]["nodes"]

    # cap
    rels = rels[:max_edges]

    node_list = []
    edge_list = []

    # Paper node
    node_list.append({
        "id": p.element_id,
        "labels": ["Paper"],
        "title": p.get("title", ""),
        "openalexId": p.get("openalexId"),
        "year": p.get("year"),
        "display": p.get("title", p.get("openalexId", "Paper")),
    })

    # Tag nodes
    for t in nodes:
        if t is None:
            continue
        labs = t.labels
        primary = labs[0] if labs else "Node"
        node_list.append({
            "id": t.element_id,
            "labels": list(labs),
            "name": t.get("name") or t.get("label") or primary,
            "display": t.get("name") or t.get("label") or primary,
        })

    # Map neo4j internal id(p/t) to element_id
    # We used id(p) in rels, but returned p/t objects; easier: rebuild edges with element_id using MATCH again
    q_edges = f"""
    MATCH (p:Paper {{openalexId:$oid}})-[r:{'|'.join(ONTO_REL_TYPES)}]->(t)
    RETURN p.openalexId AS pid, elementId(p) AS fromEid, elementId(t) AS toEid, type(r) AS type
    LIMIT $lim
    """
    edge_rows = run_cypher(q_edges, {"oid": openalex_id, "lim": max_edges})
    for e in edge_rows:
        edge_list.append({
            "from": e["fromEid"],
            "to": e["toEid"],
            "type": e["type"],
        })

    return node_list, edge_list

def render_pyvis_graph(nodes: List[Dict], edges: List[Dict], height_px: int = 520) -> None:
    if not nodes or not edges:
        st.info("그래프에 표시할 연결이 없습니다.")
        return

    net = Network(height=f"{height_px}px", width="100%", directed=True)
    net.barnes_hut()

    # add nodes
    for n in nodes:
        labels = n.get("labels", [])
        primary = labels[0] if labels else "Node"
        style = NODE_STYLE.get(primary, {"shape": "dot", "size": 16})

        title = ""
        if primary == "Paper":
            title = f"{n.get('title','')}\n{n.get('openalexId','')}"
        else:
            title = f"{primary}: {n.get('display','')}"
        net.add_node(
            n["id"],
            label=str(n.get("display", ""))[:80],
            title=title,
            shape=style.get("shape", "dot"),
            size=style.get("size", 16),
        )

    # add edges with style
    for e in edges:
        rtype = e.get("type", "REL")
        stl = REL_STYLE.get(rtype, {"width": 1})
        net.add_edge(
            e["from"],
            e["to"],
            label=rtype,
            title=rtype,
            width=stl.get("width", 1),
            arrows="to",
            dashes=stl.get("dashes", False),
        )

    # options: readable labels
    net.set_options("""
    var options = {
      "edges": {
        "smooth": { "type": "dynamic" },
        "font": { "size": 10 }
      },
      "nodes": {
        "font": { "size": 12 }
      },
      "physics": {
        "barnesHut": { "gravitationalConstant": -25000, "springLength": 120 }
      }
    }
    """)

    html = net.generate_html()
    components.html(html, height=height_px + 40, scrolling=False)

# -----------------------------
# Node-based expansion (탐색 확장)
# -----------------------------
def get_papers_by_tag(tag_label: str, tag_name: str, k: int) -> pd.DataFrame:
    q = f"""
    MATCH (t:{tag_label})
    WHERE toLower(coalesce(t.name,t.label)) = toLower($tname)
    MATCH (p:Paper)-[r:{'|'.join(ONTO_REL_TYPES)}]->(t)
    RETURN p.openalexId AS openalexId, p.title AS title, p.year AS year,
           ("via " + type(r) + " -> " + $tag_label + " / " + coalesce(t.name,t.label)) AS reason,
           1.0 AS score
    ORDER BY year DESC
    LIMIT $k
    """
    rows = run_cypher(q, {"tname": tag_name, "k": k, "tag_label": tag_label})
    if not rows:
        return pd.DataFrame(columns=["openalexId", "title", "year", "reason", "score"])
    return pd.DataFrame(rows)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="EvolBio Ontology Search", layout="wide")

st.title("SEMANTICA (Neo4j)")

# Sidebar search
st.sidebar.header("검색")
keyword = st.sidebar.text_input("키워드 (Paper title/label)", value="genomic")
k = st.sidebar.slider("결과 수", min_value=5, max_value=50, value=20, step=5)

# 3) Coverage panel
cov = get_coverage_stats()
st.subheader("데이터 커버리지 (Ontology-tagged Coverage)")
c1, c2, c3 = st.columns(3)
c1.metric("전체 Paper 수", f"{cov['total']}")
c2.metric("Ontology-tagged Paper 수", f"{cov['tagged']}")
c3.metric("Coverage", f"{cov['coverage']:.1f}%")

with st.expander("관계 타입별 커버리지 보기"):
    st.dataframe(cov["rel_df"], use_container_width=True)

st.divider()

# 1) Search results with evidence
st.subheader("검색 결과 (근거 포함)")
results_df = search_papers_with_evidence(keyword, k)

if results_df.empty:
    st.warning("검색 결과가 없습니다.")
    st.stop()

st.dataframe(
    results_df[["openalexId", "title", "year", "reason"]],
    use_container_width=True,
    hide_index=True,
)

# Evidence card: 클릭/선택용
paper_ids = results_df["openalexId"].tolist()
chosen_id = st.selectbox("Paper 선택 (openalexId)", options=paper_ids, index=0)

# 선택한 논문의 “검색 근거 카드”
chosen_row = results_df[results_df["openalexId"] == chosen_id].iloc[0]
with st.expander("선택 논문: 검색 근거 카드", expanded=True):
    st.markdown(f"선택 논문: {chosen_row['title']} ({int(chosen_row['year']) if pd.notna(chosen_row['year']) else 'N/A'})")
    st.markdown("검색 근거(reason)")
    for item in str(chosen_row["reason"]).split(" | "):
        st.write(f"- {item}")
    st.caption("※ title match / tag match(관계→온톨로지 노드)로 근거를 구분해 표시합니다.")

st.divider()

# Paper detail + tag coverage notice
st.subheader("논문 상세 / 연결된 온톨로지 태그")
detail = get_paper_detail(chosen_id)

st.markdown(f"### {detail.get('title','(no title)')} ({detail.get('year','N/A')})")
st.markdown(f"OpenAlex: {detail.get('openalexId')}")

# 태그가 하나도 없으면 안내
if (
    not detail["methods"]
    and not detail["taxa"]
    and not detail["software"]
    and not detail["assumptions"]
    and not detail["contexts"]
    and not detail["researchTasks"]
    and not detail["dataTypes"]
):
    st.info("이 논문은 아직 온톨로지 태그가 할당되지 않았습니다.")

colL, colR = st.columns(2)
with colL:
    st.write("Methods"); st.write(detail["methods"])
    st.write("Taxa"); st.write(detail["taxa"])
    st.write("Software"); st.write(detail["software"])
    st.write("DataTypes"); st.write(detail["dataTypes"])
with colR:
    st.write("Contexts"); st.write(detail["contexts"])
    st.write("Research Tasks"); st.write(detail["researchTasks"])
    st.write("Assumptions"); st.write(detail["assumptions"])

st.divider()

# Similar recommendation + evidence display
st.subheader("🔎 유사 논문 추천 (공유 온톨로지 태그 기반 + 근거 표시)")
sim_k = st.slider("추천 개수", min_value=5, max_value=30, value=10, step=5)
sim_df = get_similar_papers(chosen_id, sim_k)

if sim_df.empty:
    st.info("유사 논문이 없습니다. (공유 온톨로지 태그가 없는 경우)")
else:
    show_df = sim_df[["openalexId", "title", "year", "score"]].copy()
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    with st.expander("추천 근거(공유 태그) 보기"):
        # 상위 몇 개만 카드로
        for i, row in sim_df.head(10).iterrows():
            st.markdown(f"**{row['title']}** ({row['year']})  — score={row['score']}")
            ev = row["evidence"] or []
            # evidence를 보기 좋게 정리
            for e in ev[:20]:
                st.write(f"- {e.get('rel')} → {e.get('label')} / {e.get('name')}")
            st.markdown("---")

st.divider()

# Graph (2번: edge 강조 + legend)
st.subheader("그래프 탐색 (선택 논문 중심 1-hop)")
max_edges = st.slider("그래프 노드/엣지 최대", min_value=10, max_value=200, value=80, step=10)

nodes, edges = get_graph_1hop(chosen_id, max_edges)

st.caption("드래그: 이동 / 휠: 확대·축소 / edge 라벨: 관계(근거)")
with st.expander("관계 타입 범례(스타일)", expanded=False):
    for rel, stl in REL_STYLE.items():
        st.write(f"- {rel}: width={stl.get('width',1)}" + (" (dashed)" if stl.get("dashes") else ""))

render_pyvis_graph(nodes, edges, height_px=540)

st.divider()

# Graph-based expansion (node select)
st.subheader("그래프 노드로 탐색 확장 (노드 선택 → 관련 논문 재검색)")

# 후보 노드 목록 만들기 (Paper 제외)
tag_candidates = []
for n in nodes:
    labels = n.get("labels", [])
    if "Paper" in labels:
        continue
    primary = labels[0] if labels else None
    name = n.get("display")
    if primary and name:
        tag_candidates.append(f"{primary} | {name}")

if not tag_candidates:
    st.info("그래프에 표시할 연결 노드가 없습니다.")
else:
    chosen_tag = st.selectbox("그래프에 연결된 태그 선택", options=tag_candidates, index=0)
    tag_label, tag_name = [x.strip() for x in chosen_tag.split("|", 1)]

    expand_k = st.slider("확장 검색 결과 수", min_value=5, max_value=50, value=20, step=5)
    exp_df = get_papers_by_tag(tag_label, tag_name, expand_k)

    if exp_df.empty:
        st.info("이 태그로 연결된 논문이 없습니다.")
    else:
        st.dataframe(exp_df[["openalexId", "title", "year", "reason"]], use_container_width=True, hide_index=True)
