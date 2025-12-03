#%%
import random
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

#%%
# ----------------- 1. CONTENT EMBEDDING (TEXT FEATURE) GENERATION -----------------

def get_text_data(tx):
    """Retrieves all necessary text (abstracts) from Paper and Dataset nodes."""
    print("-> Retrieving text data for embedding...")
    # Cypher query to retrieve all Paper and Dataset nodes and their abstracts/titles
    query = """
    MATCH (n) 
    WHERE n:Paper OR n:Dataset
    RETURN n.id AS id, n.abstract AS text
    """
    result = tx.run(query)

    # Return as a dictionary: {id: text}
    return {record['id']: record['text'] for record in result}


def generate_and_store_content_embeddings(tx, node_texts):
    """
    SIMULATES generation of 768-dim embeddings and stores them back to the graph.

    In a real scenario, you would initialize the model here:
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(list(node_texts.values()))
    """
    print("-> Simulating 768-dim content embedding generation...")

    ids = list(node_texts.keys())

    # Simulated Embedding Generation: Create a list of 768 random floats for each node.
    # We use a fixed seed to ensure this simulation is reproducible across systems.
    random.seed(42)

    embeddings = [
        [random.uniform(-0.1, 0.1) for _ in range(768)]
        for _ in range(len(ids))
    ]

    # Store the embeddings back in the graph
    print("-> Storing content_embedding property on nodes...")

    for node_id, embedding_vector in zip(ids, embeddings):
        # Determine node label based on ID prefix
        label = "Paper" if node_id.startswith('P') else "Dataset"

        # Cypher query to update the node with the new property
        tx.run(f"""
        MATCH (n:{label} {{id: $id}}) 
        SET n.content_embedding = $embedding
        """, id=node_id, embedding=embedding_vector)

    print(f"-> Content embeddings generated and stored for {len(ids)} nodes.")
    return len(ids)

#%%
# ----------------- 2. COLLABORATIVE EMBEDDING (STRUCTURAL FEATURE) BASELINE -----------------

def simulate_collaborative_embeddings(tx):
    """
    In a real scenario, you would run Node2vec here to generate 128-dim embeddings
    based purely on the CITES and INTERACTS_WITH relationships.
    We are simulating this by storing a placeholder property.
    """
    print("\n-> Simulating 128-dim collaborative embedding generation (Node2vec placeholder)...")

    # Get all Research, Paper, and Dataset nodes
    query = "MATCH (n) WHERE n:Researcher OR n:Paper OR n:Dataset RETURN n.id AS id, labels(n)[0] AS label"
    nodes = [(record['id'], record['label']) for record in tx.run(query)]

    random.seed(42)  # Use the same seed for reproducibility

    for node_id, label in nodes:
        # Generate a simulated 128-dim vector
        collaborative_vector = [random.uniform(-0.5, 0.5) for _ in range(128)]

        # Update the node with the new property
        tx.run(f"""
        MATCH (n:{label} {{id: $id}})
        SET n.collaborative_embedding = $embedding
        """, id=node_id, embedding=collaborative_vector)

    print(f"-> Collaborative embeddings (128-dim) simulated and stored for {len(nodes)} nodes.")
    return len(nodes)

#%%
# ----------------- 3. BASELINE SETUP (Matrix Factorization Proxy) -----------------

def run_baseline_setup(tx):
    """
    The baseline setup is already handled by the INTERACTS_WITH relationship
    in Phase 1. This function is a placeholder for future Matrix Factorization
    training outside of the database.
    """
    print("\n-> Baseline setup complete (INTERACTS_WITH relationships are ready for MF).")

#%%
# ----------------- 4. MAIN EXECUTION BLOCK -----------------

def main():
    print("Starting Project 5 Feature Extraction Script...")

    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"\n[SUCCESS] Connected to Neo4j.")

        with driver.session() as session:
            # Step 1: Content Embeddings
            node_texts = session.execute_read(get_text_data)
            content_nodes_updated = session.execute_write(generate_and_store_content_embeddings, node_texts)

            # Step 2: Collaborative Embeddings
            collaborative_nodes_updated = session.execute_write(simulate_collaborative_embeddings)

        print(f"\n[SUMMARY] Successfully updated {content_nodes_updated} nodes with content features.")
        print(f"[SUMMARY] Successfully updated {collaborative_nodes_updated} nodes with collaborative features.")

    except Exception as e:
        print(f"\n[ERROR] Feature Extraction Failed.")
        print(f"Details: {e}")

    finally:
        if driver:
            driver.close()


#%%
main()