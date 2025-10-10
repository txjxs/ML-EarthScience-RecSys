#%%
import json
import os
import random
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NASA_KG_FILE, DATA_DIR

#%%
def simulate_data_download(filepath):
    """
    Creates a simple, structured JSON file that simulates the data we would get
    from NASA GES-DISC (Datasets) and OpenAlex (Papers/Citations).
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"-> Creating simulated data file at: {filepath}")

    # Data structure simulating the core entities and their relationships
    data = {
        "datasets": [
            {"id": "D101", "name": "Global Precipitation Forecast",
             "abstract": "Predicts worldwide rainfall using satellite data.", "cmrId": "C200-NASAGSDC"},
            {"id": "D102", "name": "Arctic Sea Ice Extent 2024",
             "abstract": "High-resolution data on melting Arctic ice coverage.", "cmrId": "C201-NASAGSDC"},
            {"id": "D103", "name": "Coastal Ocean Chlorophyll-a",
             "abstract": "Monitors near-shore marine biology and productivity.", "cmrId": "C202-NASAGSDC"},
            {"id": "D104", "name": "Tropical Cyclone Intensity Data",
             "abstract": "Historical and real-time intensity measurements for hurricanes.", "cmrId": "C203-NASAGSDC"},
        ],
        "papers": [
            {"id": "P501", "title": "Model Calibration for GPM data", "abstract": "Focuses on D101 validation.",
             "author_ids": ["R901", "R902"]},
            {"id": "P502", "title": "Analyzing Ice Trends",
             "abstract": "A study using D102 and D104 for climate modeling.", "author_ids": ["R902", "R903"]},
            {"id": "P503", "title": "Marine Biology and Coastal Mapping", "abstract": "Explores D103 applications.",
             "author_ids": ["R904"]},
            {"id": "P504", "title": "Storm Surge Prediction using Satellite Data",
             "abstract": "Advanced methods for predicting flood risk with D104.", "author_ids": ["R901", "R904"]},
        ],
        "researchers": [
            {"id": "R901", "name": "Dr. Sarah Chen"},
            {"id": "R902", "name": "Prof. Alex Singh"},
            {"id": "R903", "name": "Jia Lee"},
            {"id": "R904", "name": "Mike Davis"},
        ],
        # Simulated OpenAlex citation links (Paper -> CITES -> Paper)
        "citations": [
            {"source": "P502", "target": "P501"},
            {"source": "P504", "target": "P502"},
            {"source": "P503", "target": "P501"},
        ]
    }
    # Save the structure to the JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print("-> Data simulation successful.")
    return data
#%%
# ----------------- 2. NEO4J INGESTION LOGIC -----------------

def ingest_data_transaction(tx, data):
    """
    Clears the database and runs Cypher queries to create nodes and relationships.
    """
    print("\n--- Starting Ingestion Transaction ---")

    # Clean Slate Policy: Delete all existing nodes and relationships
    tx.run("MATCH (n) DETACH DELETE n")
    print("-> Clean slate applied: Existing data removed.")

    # 1. Create Nodes
    node_count = 0

    # Create Researchers
    for r in data['researchers']:
        tx.run("MERGE (r:Researcher {id: $id}) SET r.name = $name", id=r['id'], name=r['name'])
        node_count += 1

    # Create Papers
    for p in data['papers']:
        tx.run("MERGE (p:Paper {id: $id}) SET p.title = $title, p.abstract = $abstract",
               id=p['id'], title=p['title'], abstract=p['abstract'])
        node_count += 1

    # Create Datasets
    for d in data['datasets']:
        tx.run("MERGE (d:Dataset {id: $id}) SET d.name = $name, d.abstract = $abstract, d.cmrId = $cmrId",
               id=d['id'], name=d['name'], abstract=d['abstract'], cmrId=d['cmrId'])
        node_count += 1

    print(f"-> Nodes created: {node_count}")

    # 2. Create Relationships
    rel_count = 0

    # AUTHORED (Researcher -> Paper) and USED_IN (Paper -> Dataset)
    for paper in data['papers']:
        paper_id = paper['id']

        # Link Paper to Datasets (Simulated from metadata: D101 -> P501, D102 -> P502, etc.)
        # We assume the last letter of the paper ID maps to the last number of the dataset ID
        # E.g., P501 uses D101, P502 uses D102.
        dataset_id = "D10" + paper_id[-1]
        tx.run("MATCH (p:Paper {id: $pid}), (d:Dataset {id: $did}) "
               "MERGE (d)-[:USED_IN]->(p)", pid=paper_id, did=dataset_id)
        rel_count += 1

        # Link Researchers to Paper (AUTHORED)
        for researcher_id in paper['author_ids']:
            tx.run("MATCH (r:Researcher {id: $rid}), (p:Paper {id: $pid}) "
                   "MERGE (r)-[:AUTHORED]->(p)", rid=researcher_id, pid=paper_id)
            rel_count += 1

            # 3. Create Synthetic INTERACTS_WITH link for RecSys training (Baseline)
            # This is your Researcher-to-Dataset interaction for the Matrix Factorization baseline
            tx.run("MATCH (r:Researcher {id: $rid}), (d:Dataset {id: $did}) "
                   "MERGE (r)-[:INTERACTS_WITH {rating: $rating}]->(d)",
                   rid=researcher_id, did=dataset_id, rating=random.randint(3, 5))
            rel_count += 1

    # CITES (Paper -> CITES -> Paper)
    for citation in data['citations']:
        tx.run("MATCH (source:Paper {id: $sid}), (target:Paper {id: $tid}) "
               "MERGE (source)-[:CITES]->(target)", sid=citation['source'], tid=citation['target'])
        rel_count += 1

    print(f"-> Relationships created: {rel_count}")
    print("--- Ingestion Transaction Complete ---")
    return node_count, rel_count

#%%
def main():
    """Handles the Neo4j connection and script execution."""
    print("Starting Project 5 Setup Script...")

    # 1. Simulate data download
    data = simulate_data_download(NASA_KG_FILE)

    # 2. Connect to Neo4j
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"\n[SUCCESS] Connected to Neo4j at {NEO4J_URI}")

        # 3. Run the ingestion transaction
        with driver.session() as session:
            node_count, rel_count = session.execute_write(ingest_data_transaction, data)

        print(f"\n[SUCCESS] Graph ingestion complete. Nodes created: {node_count}. Relationships created: {rel_count}.")

    except Exception as e:
        print(f"\n[ERROR] Neo4j Connection or Ingestion Failed.")
        print(f"Check your config.py password and ensure Neo4j Desktop is running.")
        print(f"Details: {e}")

    finally:
        if driver:
            driver.close()


#%%
main()
print("\nFinished Project 5 Setup Script.")