import sys
import os
import time
import numpy as np
from neo4j import GraphDatabase

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- CONFIGURATION ---
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD

BATCH_SIZE = 1000

class AuthorEmbeddingGenerator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self):
        print("Starting Author Embedding Aggregation...")
        
        # We process authors in batches to keep memory usage low
        # We simply skip authors who already have a valid embedding (size > 1 to avoid dummy zeros if you want, 
        # but simpler to just overwrite everything to be safe).
        
        offset = 0
        
        while True:
            # 1. Fetch a batch of authors and their paper embeddings
            # We filter for papers that actually HAVE a bert embedding
            query = """
            MATCH (a:Author)
            WITH a ORDER BY a.globalId SKIP $offset LIMIT $batch_size
            MATCH (a)-[:WROTE]->(p:Publication)
            WHERE p.embedding_bert IS NOT NULL
            WITH a, collect(p.embedding_bert) as paper_vectors
            WHERE size(paper_vectors) > 0
            RETURN a.openAlexId as id, paper_vectors
            """
            
            with self.driver.session() as session:
                result = session.run(query, offset=offset, batch_size=BATCH_SIZE)
                data = [record.data() for record in result]
            
            if not data:
                print("No more authors to process.")
                break
                
            # 2. Compute Averages in Python (Numpy is fast at this)
            updates = []
            for row in data:
                # Stack vectors into a matrix and take the mean across axis 0
                vectors = np.array(row['paper_vectors'])
                mean_vector = np.mean(vectors, axis=0)
                
                # Normalize? Optional, but BERT vectors usually work best raw or normalized.
                # Let's keep it raw mean for now, FastRP handles normalization internally usually.
                
                updates.append({
                    'id': row['id'],
                    'vector': mean_vector.tolist()
                })
            
            # 3. Write back to Neo4j
            write_query = """
            UNWIND $updates as item
            MATCH (a:Author {openAlexId: item.id})
            SET a.embedding_bert = item.vector
            """
            
            with self.driver.session() as session:
                session.run(write_query, updates=updates)
            
            print(f"  Processed authors {offset} to {offset + len(data)}...")
            offset += BATCH_SIZE

if __name__ == "__main__":
    start = time.time()
    gen = AuthorEmbeddingGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        gen.run()
    finally:
        gen.close()
        print(f"Total time: {time.time() - start:.2f}s")