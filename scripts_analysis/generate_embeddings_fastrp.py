import sys
import os
import time
from graphdatascience import GraphDataScience

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- CONFIGURATION ---
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD

class EmbeddingGenerator:
    def __init__(self, uri, user, password):
        self.gds = GraphDataScience(uri, auth=(user, password))

    def run_content_embedding(self):
        """ Model A: Content-Driven (Metadata Graph) """
        PROJ_NAME = "proj_content_fastrp"
        print(f"\n--- Generating CONTENT Embeddings ({PROJ_NAME}) ---")

        try:
            self.gds.graph.drop(self.gds.graph.get(PROJ_NAME))
        except:
            pass

        print("Projecting Metadata Graph...")
        G, res = self.gds.graph.project(
            PROJ_NAME,
            ["Dataset", "ScienceKeyword", "Instrument", "Platform"],
            ["HAS_SCIENCEKEYWORD", "HAS_INSTRUMENT", "HAS_PLATFORM"]
        )
        
        # Content graph is small (11k nodes), so we can just write directly
        print("Running FastRP (Direct Write)...")
        self.gds.fastRP.write(
            G,
            embeddingDimension=128,
            iterationWeights=[0.8, 1.0, 1.0], 
            writeProperty="embedding_fastrp_content",
            randomSeed=42
        )
        print("Success.")
        self.gds.graph.drop(G)

    def run_collaborative_embedding(self):
        """ Model B: Collaborative (Interaction Graph) """
        PROJ_NAME = "proj_collab_fastrp"
        print(f"\n--- Generating COLLABORATIVE Embeddings ({PROJ_NAME}) ---")

        try:
            self.gds.graph.drop(self.gds.graph.get(PROJ_NAME))
        except:
            pass

        print("Projecting Interaction Graph...")
        G, res = self.gds.graph.project(
            PROJ_NAME,
            ["Dataset", "Publication", "Author"],
            {
                "USES_DATASET": {"orientation": "UNDIRECTED"},
                "WROTE": {"orientation": "UNDIRECTED"}
            }
        )

        # --- THE MEMORY FIX ---
        # Step 1: Calculate in RAM (Mutate)
        # concurrency=1 reduces peak memory usage during calculation
        print("Step 1: Calculating Vectors in RAM (Mutate)...")
        self.gds.fastRP.mutate(
            G,
            embeddingDimension=128,
            iterationWeights=[0.8, 1.0, 1.0, 1.0], 
            mutateProperty="embedding_fastrp_collab",
            randomSeed=42,
            concurrency=1  
        )
        
        # Step 2: Write to Disk using explicit batching
        # This function streams data slowly to the disk, avoiding the OOM crash
        print("Step 2: Streaming Vectors to Disk...")
        self.gds.graph.writeNodeProperties(
            G, 
            ["embedding_fastrp_collab"], 
            concurrency=1
        )
        print("Success! Collaborative embeddings saved.")

        self.gds.graph.drop(G)

    def sanity_check(self):
        print("\n--- Sanity Check ---")
        query = """
        MATCH (d:Dataset) 
        WHERE d.embedding_fastrp_content IS NOT NULL 
        RETURN d.globalId, size(d.embedding_fastrp_content) as Size 
        LIMIT 1
        """
        print(self.gds.run_cypher(query))

if __name__ == "__main__":
    start = time.time()
    gen = EmbeddingGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        gen.run_content_embedding()
        gen.run_collaborative_embedding()
        gen.sanity_check()
    finally:
        print(f"Done in {time.time() - start:.2f} seconds.")