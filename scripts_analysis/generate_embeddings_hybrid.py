import sys
import os
import time
from graphdatascience import GraphDataScience

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD

class HybridEmbeddingGenerator:
    def __init__(self, uri, user, password):
        self.gds = GraphDataScience(uri, auth=(user, password))

    def run_hybrid_embedding(self):
        """
        Model 2: Hybrid (FastRP + BERT)
        Includes Authors, Publications, and Datasets.
        """
        PROJ_NAME = "proj_hybrid_full"
        print(f"\n--- Generating FULL HYBRID Embeddings ({PROJ_NAME}) ---")

        # 1. Cleanup old projection
        try:
            self.gds.graph.drop(self.gds.graph.get(PROJ_NAME))
        except:
            pass

        # 2. Project Graph WITH BERT properties
        # We now include Authors because you successfully generated their vectors!
        print("Projecting Graph (Authors + Pubs + Datasets)...")
        
        node_projection = {
            "Dataset": {
                "label": "Dataset",
                "properties": ["embedding_bert"]
            },
            "Publication": {
                "label": "Publication",
                "properties": ["embedding_bert"]
            },
            "Author": {
                "label": "Author",
                "properties": ["embedding_bert"]
            }
        }

        rel_projection = {
            "USES_DATASET": {"orientation": "UNDIRECTED"},
            "WROTE": {"orientation": "UNDIRECTED"}
        }

        # readConcurrency=1 helps prevent OOM on the MacBook during loading
        G, res = self.gds.graph.project(
            PROJ_NAME, 
            node_projection, 
            rel_projection, 
            readConcurrency=1
        )
        print(f"Projected {res['nodeCount']} nodes.")

        # 3. Run FastRP with BERT Injection
        # propertyRatio=0.5 -> 50% BERT / 50% Citation Graph
        print("Running FastRP (Mutate)...")
        self.gds.fastRP.mutate(
            G,
            embeddingDimension=128,
            iterationWeights=[0.8, 1.0, 1.0, 1.0], 
            featureProperties=["embedding_bert"], 
            propertyRatio=0.5, 
            mutateProperty="embedding_hybrid",
            randomSeed=42,
            concurrency=1 
        )
        
        # 4. Write to Disk
        print("Writing 'embedding_hybrid' to disk...")
        self.gds.graph.writeNodeProperties(
            G, 
            ["embedding_hybrid"], 
            concurrency=1
        )
        print("Success! Full Hybrid embeddings saved.")

        self.gds.graph.drop(G)

    def sanity_check(self):
        print("\n--- Sanity Check ---")
        query = """
        MATCH (a:Author) 
        WHERE a.embedding_hybrid IS NOT NULL 
        RETURN a.openAlexId, size(a.embedding_hybrid) as Size 
        LIMIT 1
        """
        print(self.gds.run_cypher(query))

if __name__ == "__main__":
    start = time.time()
    gen = HybridEmbeddingGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        gen.run_hybrid_embedding()
        gen.sanity_check()
    finally:
        print(f"Done in {time.time() - start:.2f} seconds.")