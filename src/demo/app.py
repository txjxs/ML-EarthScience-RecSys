import streamlit as st
import numpy as np
import os
import sys
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- PATH SETUP ---
# Points to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config
from src.models.gnn import ContentSAGE

# --- CONFIGURATION ---
st.set_page_config(page_title="NASA Dataset Recommender", layout="wide")

# --- CACHED RESOURCES ---
@st.cache_resource
def init_driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

@st.cache_resource
def load_bert():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_gnn_model():
    """Load the trained GraphSAGE model to translate queries."""
    model = ContentSAGE(in_channels=384, hidden_channels=256, out_channels=128)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'content_sage.pt'))
    
    # Map to CPU ensures it runs on any machine (even without CUDA/MPS)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_datasets(_driver):
    """Load ALL datasets into RAM for instant ranking."""
    query = """
    MATCH (d:Dataset)
    RETURN d.globalId as id, d.shortName as name, d.longName as desc, 
           d.embedding_gnn_content as content_vec,
           d.embedding_gnn_collab as collab_vec
    """
    datasets = []
    with _driver.session() as session:
        result = session.run(query)
        for record in result:
            datasets.append({
                'id': record['id'],
                'name': record['name'],
                'desc': record['desc'] if record['desc'] else "No description.",
                # Convert to Numpy Arrays for fast math
                'content_vec': np.array(record['content_vec']) if record['content_vec'] else None,
                'collab_vec': np.array(record['collab_vec']) if record['collab_vec'] else None
            })
    return datasets

@st.cache_data
def get_demo_authors(_driver):
    """Get 5 real authors for the dropdown."""
    query = """
    MATCH (a:Author)
    WHERE a.embedding_gnn_collab IS NOT NULL
    RETURN a.name as name, a.embedding_gnn_collab as vector
    LIMIT 5
    """
    authors = {}
    with _driver.session() as session:
        result = session.run(query)
        for record in result:
            authors[record['name']] = np.array(record['vector'])
    return authors

# --- MAIN APP LOGIC ---

def main():
    st.title("🌍 Earth Science Dataset Recommender")
    st.markdown("Recommended based on **User History (Who you are)** + **Current Query (What you want)**.")

    # --- 1. LOAD RESOURCES ---
    with st.spinner("Connecting to Database & Loading Models..."):
        driver = init_driver()
        bert_model = load_bert()
        gnn_model = load_gnn_model()
        all_datasets = load_datasets(driver)
        demo_authors = get_demo_authors(driver)

    # --- 2. SIDEBAR (DEFINE AUTHOR FIRST) ---
    st.sidebar.header("👤 User Profile")
    author_options = ["New / Guest User"] + list(demo_authors.keys())
    
    # THIS LINE MUST COME BEFORE THE DEBUGGER
    selected_author_name = st.sidebar.selectbox("Select Researcher:", author_options)

    author_vector = None
    if selected_author_name != "New / Guest User":
        author_vector = demo_authors[selected_author_name]
        st.sidebar.success(f"Logged in as: **{selected_author_name}**")
    
    # --- 3. DEBUG SECTION (USE AUTHOR AFTER DEFINITION) ---
    with st.sidebar.expander("🛠 Developer Debugger"):
        st.write("Check what the Collaborative Model knows about this author.")
        if st.button("Show Top 5 History Items"):
            if selected_author_name == "New / Guest User":
                st.error("Select an author first.")
            else:
                vec = demo_authors[selected_author_name]
                debug_scores = []
                for d in all_datasets:
                    if d['collab_vec'] is not None:
                        sim = cosine_similarity([vec], [d['collab_vec']])[0][0]
                        debug_scores.append((d['name'], sim))
                
                debug_scores.sort(key=lambda x: x[1], reverse=True)
                
                st.write("**Top Known Preferences (LightGCN):**")
                for name, score in debug_scores[:5]:
                    st.write(f"- {name}: `{score:.4f}`")

    # --- 4. MAIN INPUT ---
    query = st.text_area("🔎 Abstract / Search Query:", 
                         placeholder="Paste abstract here...",
                         height=100)

    results = []
    mode = ""

    # --- DECISION ENGINE ---
    
    def project_query(text_query):
        """Converts Text -> BERT (384) -> GNN (128)"""
        raw_vec = bert_model.encode(text_query)
        t_vec = torch.tensor(raw_vec).unsqueeze(0) 
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        with torch.no_grad():
            gnn_vec = gnn_model(t_vec, empty_edge_index).numpy()[0]
        return gnn_vec

    # CASE 1: HYBRID (Author + Query)
    if query.strip() and author_vector is not None:
        mode = "Hybrid (History + Intent)"
        st.subheader(f"Personalized Results for **{selected_author_name}** matching query")
        st.info("🚀 **Mode: Hybrid** — Combining LightGCN (History) and GraphSAGE (Topic).")
        
        query_vec = project_query(query)
        
        scores = []
        for d in all_datasets:
            sim_content = 0.0
            if d['content_vec'] is not None:
                sim_content = cosine_similarity([query_vec], [d['content_vec']])[0][0]
                # FIX 1: Clamp Negative Content Scores
                sim_content = max(0.0, sim_content)
            
            sim_collab = 0.0
            if d['collab_vec'] is not None:
                sim_collab = cosine_similarity([author_vector], [d['collab_vec']])[0][0]
            
            # FIX 2: Weighted Fusion based on History Strength
            # if sim_collab > 0.99:
            #     # Strong History? Trust it more (80%)
            #     final_score = (0.4 * sim_content) + (0.6 * sim_collab)
            # else:
            #     # No History? Trust Content more (80%)
            #     final_score = (0.6 * sim_content) + (0.4 * sim_collab)
            final_score = (sim_content + sim_collab) / 2
            scores.append((d, final_score, sim_content, sim_collab))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:5]

    # CASE 2: QUERY ONLY (New User)
    elif query.strip():
        mode = "Content-Based (GraphSAGE)"
        st.subheader(f"Results for query")
        st.warning("**Mode: Content-Only** — No user history found. Matching text topic only.")
        
        query_vec = project_query(query)
        
        scores = []
        for d in all_datasets:
            if d['content_vec'] is not None:
                sim = cosine_similarity([query_vec], [d['content_vec']])[0][0]
                scores.append((d, sim, sim, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:5]

    # CASE 3: AUTHOR ONLY (Profile Page)
    elif author_vector is not None:
        mode = "Collaborative (LightGCN)"
        st.subheader(f"Recommended for **{selected_author_name}**")
        st.success("**Mode: Collaborative** — Based purely on your citation history.")
        
        scores = []
        for d in all_datasets:
            if d['collab_vec'] is not None:
                sim = cosine_similarity([author_vector], [d['collab_vec']])[0][0]
                scores.append((d, sim, 0.0, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:5]

    else:
        st.write("👈 **Start by selecting a Researcher or typing a Query.**")
        st.stop()

    # --- DISPLAY RESULTS ---
    st.markdown("---")
    for i, (dataset, final, content_s, collab_s) in enumerate(results):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric(label="Match Score", value=f"{final:.1%}")
                if mode.startswith("Hybrid"):
                    st.caption(f"Topic: {content_s:.2f} | Hist: {collab_s:.2f}")
            with col2:
                st.markdown(f"### {i+1}. {dataset['name']}")
                st.caption(dataset['desc'][:300] + "...")
                
                tags = []
                if content_s > 0.2: tags.append("`Topic Match`")
                if collab_s > 0.2: tags.append("`History Match`")
                if tags: st.markdown(" ".join(tags))
            st.divider()

if __name__ == "__main__":
    main()