import sys
import os
import time
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

# --- CONFIGURATION ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD

# Settings for consistent evaluation
RANDOM_SEED = 42
K = 10                  # Top-K for metrics
SAMPLE_SIZE = 2000      # How many users/papers to test

class UnifiedEvaluator:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.results = []
        
        # Set deterministic seeds
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def close(self):
        self.driver.close()

    def _compute_metrics(self, vectors_a, vectors_b, ground_truth, task_name, model_name):
        """
        Generic engine to calculate Precision, Recall, and MRR.
        """
        if not vectors_a:
            print(f"  [WARN] No data found for {model_name}. Skipping.")
            return

        print(f"  -> Evaluating {model_name} on {len(vectors_a)} samples...")
        
        # Prepare Matrices
        target_ids = sorted(list(vectors_b.keys())) 
        target_matrix = np.array([vectors_b[tid] for tid in target_ids])
        source_matrix = np.array(vectors_a)

        # Compute Similarity (Cosine)
        scores = cosine_similarity(source_matrix, target_matrix)

        precisions, recalls, mrrs = [], [], []

        for i, true_items in enumerate(ground_truth):
            # Get Top-K indices
            top_k_idx = np.argpartition(-scores[i], K)[:K]
            top_k_idx = top_k_idx[np.argsort(-scores[i][top_k_idx])]
            
            recommendations = [target_ids[idx] for idx in top_k_idx]

            hits = 0
            first_hit_rank = 0
            
            for rank, dataset_id in enumerate(recommendations):
                if dataset_id in true_items:
                    hits += 1
                    if first_hit_rank == 0:
                        first_hit_rank = rank + 1
            
            precisions.append(hits / K)
            recalls.append(hits / len(true_items) if len(true_items) > 0 else 0)
            mrrs.append(1.0 / first_hit_rank if first_hit_rank > 0 else 0)

        avg_p = np.mean(precisions)
        avg_r = np.mean(recalls)
        avg_mrr = np.mean(mrrs)
        
        self.results.append({
            "Task": task_name,
            "Model": model_name,
            "MRR@10": avg_mrr,
            "Precision@10": avg_p,
            "Recall@10": avg_r
        })
        
        print(f"     Results: MRR={avg_mrr:.4f}, P={avg_p:.4f}, R={avg_r:.4f}")

    # ==========================================
    # TASK 1: AUTHOR PREDICTION (Personalized)
    # ==========================================
    def evaluate_author_task(self):
        print("\n=== TASK 1: Personalized Recommendation (Author -> Dataset) ===")
        print("Fetching Ground Truth for Authors...")
        
        # Removed the comment "-- Deterministic ordering" to fix Syntax Error
        query = f"""
        MATCH (a:Author)-[:WROTE]->(:Publication)-[:USES_DATASET]->(d:Dataset)
        WHERE a.embedding_bert IS NOT NULL 
        WITH a, collect(DISTINCT d.globalId) as truth
        RETURN a.openAlexId as id, truth
        ORDER BY a.openAlexId
        LIMIT {SAMPLE_SIZE}
        """
        
        with self.driver.session() as session:
            res = session.run(query)
            data = [r.data() for r in res]
            
        if not data:
            print("No authors found! Run ingestion scripts first.")
            return

        test_ids = [r['id'] for r in data]
        ground_truth = [set(r['truth']) for r in data]
        
        models = [
            ("Zero-Shot (Raw BERT)", "embedding_bert"),
            ("Baseline (FastRP)", "embedding_fastrp_collab"),
            ("Collaborative (LightGCN)", "embedding_gnn_collab")
        ]
        
        for pretty_name, property_key in models:
            self._run_author_model(test_ids, ground_truth, pretty_name, property_key)

    def _run_author_model(self, author_ids, ground_truth, model_name, property_key):
        with self.driver.session() as session:
            # Fetch Authors
            q_auth = f"""
            MATCH (a:Author)
            WHERE a.openAlexId IN $ids AND a.{property_key} IS NOT NULL
            RETURN a.openAlexId as id, a.{property_key} as vector
            """
            res_a = session.run(q_auth, ids=author_ids)
            auth_map = {r['id']: np.array(r['vector']) for r in res_a}
            
            # Fetch Datasets
            q_data = f"""
            MATCH (d:Dataset)
            WHERE d.{property_key} IS NOT NULL
            RETURN d.globalId as id, d.{property_key} as vector
            """
            res_d = session.run(q_data)
            dataset_vectors = {r['id']: np.array(r['vector']) for r in res_d}
            
        aligned_vectors = []
        valid_indices = []
        
        for i, aid in enumerate(author_ids):
            if aid in auth_map:
                aligned_vectors.append(auth_map[aid])
                valid_indices.append(i)
        
        if not aligned_vectors:
            print(f"  [WARN] {model_name} has no vectors for these authors.")
            return

        filtered_truth = [ground_truth[i] for i in valid_indices]
        
        self._compute_metrics(aligned_vectors, dataset_vectors, filtered_truth, "Author Rec", model_name)

    # ==========================================
    # TASK 2: CONTENT DISCOVERY (Paper -> Dataset)
    # ==========================================
    def evaluate_paper_task(self):
        print("\n=== TASK 2: Content Discovery (Paper -> Dataset) ===")
        print("Fetching Ground Truth for Papers...")
        
        query = f"""
        MATCH (p:Publication)-[:USES_DATASET]->(d:Dataset)
        WHERE p.embedding_bert IS NOT NULL 
        WITH p, collect(DISTINCT d.globalId) as truth
        RETURN p.globalId as id, truth
        ORDER BY p.globalId
        LIMIT {SAMPLE_SIZE}
        """
        
        with self.driver.session() as session:
            res = session.run(query)
            data = [r.data() for r in res]
            
        test_ids = [r['id'] for r in data]
        ground_truth = [set(r['truth']) for r in data]
        
        models = [
            ("Raw Content (BERT)", "embedding_bert"),
            ("Content GNN (GraphSAGE)", "embedding_gnn_content")
        ]
        
        for pretty_name, property_key in models:
            self._run_paper_model(test_ids, ground_truth, pretty_name, property_key)

    def _run_paper_model(self, paper_ids, ground_truth, model_name, property_key):
        with self.driver.session() as session:
            # Fetch Papers
            q_paper = f"""
            MATCH (p:Publication)
            WHERE p.globalId IN $ids AND p.{property_key} IS NOT NULL
            RETURN p.globalId as id, p.{property_key} as vector
            """
            res_p = session.run(q_paper, ids=paper_ids)
            paper_map = {r['id']: np.array(r['vector']) for r in res_p}
            
            # Fetch Datasets
            q_data = f"""
            MATCH (d:Dataset)
            WHERE d.{property_key} IS NOT NULL
            RETURN d.globalId as id, d.{property_key} as vector
            """
            res_d = session.run(q_data)
            dataset_vectors = {r['id']: np.array(r['vector']) for r in res_d}

        aligned_vectors = []
        valid_indices = []
        
        for i, pid in enumerate(paper_ids):
            if pid in paper_map:
                aligned_vectors.append(paper_map[pid])
                valid_indices.append(i)
                
        filtered_truth = [ground_truth[i] for i in valid_indices]
        
        self._compute_metrics(aligned_vectors, dataset_vectors, filtered_truth, "Content Rec", model_name)

    def print_leaderboard(self):
        print("\n" + "="*70)
        print(f"{'TASK':<15} | {'MODEL':<25} | {'MRR@10':<8} | {'PREC':<8} | {'RECALL':<8}")
        print("-" * 70)
        for r in self.results:
            print(f"{r['Task']:<15} | {r['Model']:<25} | {r['MRR@10']:.4f}   | {r['Precision@10']:.4f}   | {r['Recall@10']:.4f}")
        print("="*70)

if __name__ == "__main__":
    evaluator = UnifiedEvaluator()
    try:
        evaluator.evaluate_author_task()
        evaluator.evaluate_paper_task()
    finally:
        evaluator.close()
    
    evaluator.print_leaderboard()