import os
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

# ----------------------------- 1. 접속 정보 및 실제 DB 스키마 -----------------------------
# [필독] image_208455에서 확인된 'Unauthorized' 에러를 피하려면 비밀번호가 정확해야 합니다.
NEO4J_URI = "neo4j://127.0.0.1:7687" 
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!" 

# 실제 DB 노드 라벨 및 관계 타입 동기화
ONTO_NODE_LABELS = [
    "Assumption", "Context", "ContributionType", "DataType", 
    "EnvironmentType", "GeoScope", "Method", "ResearchTask", "Software", "Taxon"
]
ONTO_REL_TYPES = [
    "HAS_CONTRIBUTION", "HAS_ENVIRONMENT", "HAS_GEOSCOPE", "HAS_RESEARCH_TASK", 
    "STUDIES_TAXON", "USES_DATATYPE", "USES_METHOD", "USES_SOFTWARE"
]

# ----------------------------- 2. 스타일 설정 (가독성 & 깨짐 방지) -----------------------------
st.set_page_config(page_title="SEMANTICA - Ontology Explorer", layout="wide")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { max-width: 1200px; margin: 0 auto; }
    /* 연도가 세로로 깨지는 현상 방지 (image_2025fe 해결) */
    [data-testid="stDataFrame"] td { white-space: nowrap !important; }
    .stMetric { background-color: #F8F9FA; border: 1px solid #E3E8EF; border-radius: 0.5rem; padding: 1rem; }
    h1 { font-size: 2.5rem !important; font-weight: 800 !important; margin: 0 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------- 3. 안전한 데이터베이스 함수 -----------------------------
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def run_cypher(query, params=None):
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [r.data() for r in result]
    except Exception as e:
        # image_208016에서 확인된 접속 실패 시 IndexError 방지
        st.sidebar.error(f"❌ DB 접속 오류: {e}")
        return []

def get_coverage_stats():
    total_res = run_cypher("MATCH (p:Paper) RETURN count(p) AS total")
    total = total_res[0]["total"] if total_res else 0
    rel_types_str = '|'.join(ONTO_REL_TYPES)
    tagged_res = run_cypher(f"MATCH (p:Paper) WHERE EXISTS {{ MATCH (p)-[:{rel_types_str}]->() }} RETURN count(p) AS tagged")
    tagged = tagged_res[0]["tagged"] if tagged_res else 0
    coverage = (tagged / total * 100.0) if total > 0 else 0.0
    return {"total": total, "tagged": tagged, "coverage": coverage}

# ----------------------------- 4. 핵심 로직 (TypeError 수정 완료) -----------------------------
def search_papers(keyword, k):
    """다중 키워드 검색 및 NoneType 에러 수정 버전"""
    raw_kw = keyword.strip()
    if not raw_kw: return pd.DataFrame()
    
    keywords = [kw.lower() for kw in raw_kw.split() if kw]
    rel_types_str = '|'.join(ONTO_REL_TYPES)
    
    # Cypher: NULL 값 방지 및 태그 매칭 고도화
    q = f"""
    MATCH (p:Paper)
    OPTIONAL MATCH (p)-[r:{rel_types_str}]->(t)
    WITH p, 
         collect(DISTINCT toLower(p.title)) + collect(DISTINCT toLower(coalesce(t.name, t.label, ""))) AS all_text,
         collect(DISTINCT {{type: type(r), name: coalesce(t.name, t.label, "Unknown")}}) AS tags
    
    WHERE all(kw IN $kws WHERE any(txt IN all_text WHERE txt CONTAINS kw))
    
    RETURN p.openalexId AS id, p.title AS title, p.year AS year,
           [tag IN tags WHERE tag.type IS NOT NULL | tag.type + ": " + tag.name] AS evidence
    ORDER BY p.year DESC
    LIMIT $limit
    """
    
    rows = run_cypher(q, {"kws": keywords, "limit": k})
    if not rows: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    # [수정] TypeError 방지를 위해 문자열 인스턴스인지 한 번 더 검사
    df['reason'] = df['evidence'].apply(lambda x: " | ".join([str(i) for i in x if i is not None][:3]))
    return df

def get_paper_detail(openalex_id):
    q = """
    MATCH (p:Paper {openalexId: $oid})
    OPTIONAL MATCH (p)-[r]->(t)
    WHERE any(lbl IN labels(t) WHERE lbl IN $labels)
    RETURN labels(t)[0] AS type, collect(DISTINCT coalesce(t.name, t.label, "Not Specified")) AS names
    """
    rows = run_cypher(q, {"oid": openalex_id, "labels": ONTO_NODE_LABELS})
    detail = {label: [] for label in ONTO_NODE_LABELS}
    for r in rows:
        if r['type'] in detail: detail[r['type']] = r['names']
    return detail

def get_similar_papers(openalex_id, k=5):
    rel_types_str = '|'.join(ONTO_REL_TYPES)
    q = f"""
    MATCH (p:Paper {{openalexId: $oid}})-[:{rel_types_str}]->(t)
    WITH p, collect(id(t)) AS target_tags
    MATCH (p2:Paper)-[:{rel_types_str}]->(t2)
    WHERE p2 <> p AND id(t2) IN target_tags
    RETURN p2.title AS title, p2.year AS year, count(DISTINCT t2) AS common_tags
    ORDER BY common_tags DESC LIMIT $k
    """
    rows = run_cypher(q, {"oid": openalex_id, "k": k})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 연도 정수형 고정 (image_2025fe 해결)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    return df

def render_graph(openalex_id, max_edges=80):
    rel_types_str = '|'.join(ONTO_REL_TYPES)
    q = f"""
    MATCH (p:Paper {{openalexId: $oid}})
    OPTIONAL MATCH (p)-[r:{rel_types_str}]->(t)
    RETURN p.title AS p_title, elementId(p) AS p_id, 
           collect({{t_id: elementId(t), t_name: coalesce(t.name, t.label, "Unknown"), t_type: labels(t)[0], r_type: type(r)}})[..$limit] AS rels
    """
    rows = run_cypher(q, {"oid": openalex_id, "limit": max_edges})
    if not rows or not rows[0]['p_id']: return
    
    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff")
    row = rows[0]
    net.add_node(row['p_id'], label=str(row['p_title'])[:40]+"...", title=row['p_title'], shape="dot", color="#000000", size=35)
    for rel in row['rels']:
        if rel['t_id']:
            net.add_node(rel['t_id'], label=rel['t_name'], title=rel['t_type'], shape="dot", size=22, color="#626D7D")
            net.add_edge(row['p_id'], rel['t_id'], label=rel['r_type'], color="#E3E8EF")
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -20000}}}')
    components.html(net.generate_html(), height=620)

# ----------------------------- 5. UI 실행 -----------------------------
st.markdown("""
<div style="background-color: #000; padding: 2rem; border-radius: 0.5rem; color: white; margin-bottom: 2rem;">
    <h1>SEMANTICA</h1>
    <p style="opacity: 0.8; margin:0;">AI-Powered Evolutionary Biology Ontology Explorer</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 Search Settings")
    keyword = st.text_input("검색어 입력", "genomic")
    k_limit = st.slider("최대 결과 수", 5, 50, 20)
    st.divider()
    stats = get_coverage_stats()
    st.metric("DB Coverage", f"{stats['coverage']:.1f}%")
    st.write(f"Total Papers: **{stats['total']}**")
    st.write(f"Tagged Papers: **{stats['tagged']}**")

results = search_papers(keyword, k_limit)

if results.empty:
    st.warning("⚠️ 일치하는 데이터를 찾을 수 없습니다.")
else:
    st.subheader(f"Results for '{keyword}'")
    st.dataframe(results[["title", "year", "reason"]].rename(columns={"title":"Title", "year":"Year", "reason":"Evidence"}), 
                 use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📊 Paper Deep-Dive Analysis")
    selected_title = st.selectbox("분석할 논문을 선택하세요:", results["title"].tolist())
    selected_id = results[results["title"] == selected_title]["id"].values[0]
    
    st.markdown("### 🏷️ Ontology Tags")
    details = get_paper_detail(selected_id)
    tag_cols = st.columns(4)
    active_labels = [(l, n) for l, n in details.items() if n]
    for i, (label, names) in enumerate(active_labels):
        with tag_cols[i % 4]:
            with st.expander(f"**{label}** ({len(names)})", expanded=True):
                for n in names: st.caption(f"• {n}")

    st.divider()
    st.markdown("### 🔗 Similar Papers (Shared Tags)")
    sim_df = get_similar_papers(selected_id)
    if not sim_df.empty:
        st.dataframe(sim_df.rename(columns={"title":"Title", "year":"Year", "common_tags":"Shared Tags"}), 
                     use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🕸️ Knowledge Network Graph")
    render_graph(selected_id)

st.divider()
st.caption("SEMANTICA v1.7 | TypeError & Formatting Fix Applied")