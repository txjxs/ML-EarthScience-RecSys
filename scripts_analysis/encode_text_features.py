import sys
import os
import time
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- CONFIGURATION ---
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD

# A small, fast, high-quality model for semantic similarity
MODEL_NAME = 'all-MiniLM-L6-v2' 

class TextEncoder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"Loading Sentence Transformer: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)

    def close(self):
        self.driver.close()

    def fetch_nodes(self, label):
        """
        Fetches ID and Text (Title + Abstract) for a specific label.
        """
        print(f"Fetching {label} nodes...")
        query = f"""
        MATCH (n:{label})
        WHERE n.title IS NOT NULL OR n.abstract IS NOT NULL
        RETURN n.globalId as id, 
               coalesce(n.title, '') + ' ' + coalesce(n.abstract, '') as text
        """
        with self.driver.session() as session:
            result = session.run(query)
            # Return list of dicts: [{'id': '...', 'text': '...'}]
            return [record.data() for record in result]

    def update_embeddings(self, label, data_batch):
        """
        Encodes text and writes to Neo4j in batches.
        """
        # 1. Encode text to vectors
        texts = [item['text'] for item in data_batch]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # 2. Prepare payload
        payload = []
        for i, item in enumerate(data_batch):
            payload.append({
                'id': item['id'],
                'vector': embeddings[i].tolist() # Convert numpy to list for Neo4j
            })

        # 3. Write to DB
        query = f"""
        UNWIND $batch as item
        MATCH (n:{label} {{globalId: item.id}})
        SET n.embedding_bert = item.vector
        """
        with self.driver.session() as session:
            session.run(query, batch=payload)

    def run(self):
        # We need embeddings for Datasets AND Publications for the Hybrid model
        for label in ["Dataset", "Publication"]:
            nodes = self.fetch_nodes(label)
            print(f"Found {len(nodes)} {label}s to encode.")
            
            BATCH_SIZE = 500
            total = len(nodes)
            
            for i in range(0, total, BATCH_SIZE):
                batch = nodes[i : i + BATCH_SIZE]
                self.update_embeddings(label, batch)
                
                if (i // BATCH_SIZE) % 10 == 0:
                    print(f"  Encoded {i}/{total}...")
            
            print(f"Finished {label}s.\n")

if __name__ == "__main__":
    start = time.time()
    encoder = TextEncoder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        encoder.run()
    finally:
        encoder.close()
        print(f"Total time: {time.time() - start:.2f}s")