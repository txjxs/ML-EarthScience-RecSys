import config
import os
import json
from neo4j import GraphDatabase
from datasets import load_dataset
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. DATA FETCHING (FROM HUGGING FACE) ---

def fetch_nasa_kg():
    """
    Downloads the NASA KG from Hugging Face and saves it as a JSONL file
    in our data directory. If the file already exists, it skips downloading.
    """
    # Use the file path from our config.py
    local_kg_path = os.path.join(config.DATA_DIR, config.NASA_KG_FILE)

    if os.path.exists(local_kg_path):
        logging.info(f"'{local_kg_path}' already exists. Skipping download.")
        return local_kg_path

    logging.info("Downloading NASA GES-DISC Knowledge Graph from Hugging Face...")
    try:
        # Load the dataset from Hugging Face
        # The project PDF (final_project_ideas.pdf) gives us the exact name
        dataset = load_dataset("nasa-gesdisc/nasa-eo-knowledge-graph")

        # The dataset is in the 'train' split. We will save it as a JSONL
        # file (JSON lines) for easier processing.
        os.makedirs(config.DATA_DIR, exist_ok=True)

        # Access the 'train' split
        train_split = dataset['train']

        with open(local_kg_path, 'w', encoding='utf-8') as f:
            for entry in train_split:
                # Ensure data is serializable
                serializable_entry = {k: v for k, v in entry.items() if v is not None}
                f.write(json.dumps(serializable_entry) + '\n')

        logging.info(f"Successfully downloaded and saved data to '{local_kg_path}'")
        return local_kg_path

    except Exception as e:
        logging.error(f"Failed to download or save dataset: {e}")
        logging.exception(e)  # More detailed error
        return None


# --- 2. NEO4J INGESTION ---

def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        logging.info("Neo4j connection successful.")
        return driver
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        return None


def ingest_data_transaction(tx, data_chunk):
    """
    The actual Neo4j transaction function to load a CHUNK of data.
    This creates Dataset and Publication nodes and the USED_IN relationship.
    """
    # This Cypher query is optimized for bulk loading.
    # It uses UNWIND to process a list of data rows at once.
    query = """
    UNWIND $rows AS row

    // Create or merge the Dataset node
    MERGE (d:Dataset {cmrId: row.dataset_cmr_id})
    ON CREATE SET
        d.shortName = row.dataset_short_name,
        d.longName = row.dataset_long_name,
        d.conceptId = row.dataset_concept_id,
        d.description = row.dataset_description

    // Create or merge the Publication (Paper) node
    MERGE (p:Paper {doi: row.publication_doi})
    ON CREATE SET
        p.title = row.publication_title,
        p.publicationDate = row.publication_date,
        p.abstract = row.publication_abstract

    // Create the relationship between them
    MERGE (d)-[r:USED_IN]->(p)
    """
    try:
        tx.run(query, rows=data_chunk)
    except Exception as e:
        logging.error(f"Error during transaction: {e}")
        raise  # Re-raise the exception to trigger a rollback


def clear_database(driver):
    """WIPES all nodes and relationships from the database."""
    logging.warning("WIPING all data from the Neo4j database...")
    try:
        with driver.session() as session:
            # Use write_transaction for this operation
            session.write_transaction(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        logging.info("Database successfully cleared.")
    except Exception as e:
        logging.error(f"Failed to clear database: {e}")


def main():
    # Step 1: Get the real data
    local_data_file = fetch_nasa_kg()
    if not local_data_file:
        logging.error("Halting execution: data file not available.")
        return

    # Step 2: Connect to Neo4j
    driver = get_neo4j_driver()
    if not driver:
        logging.error("Halting execution: Neo4j driver not available.")
        return

    # Step 3: Clear the database of old (simulated) data
    clear_database(driver)

    # Step 4: Load data in chunks for efficiency
    chunk_size = 500
    data_chunk = []
    total_loaded = 0
    total_skipped = 0

    logging.info(f"Starting ingestion from '{local_data_file}'...")
    try:
        with open(local_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # The KG file links datasets and publications.
                    # We just need to extract the relevant fields.
                    prepared_row = {
                        "dataset_cmr_id": data.get("dataset_cmr_id"),
                        "dataset_short_name": data.get("dataset_short_name"),
                        "dataset_long_name": data.get("dataset_long_name"),
                        "dataset_concept_id": data.get("dataset_concept_id"),
                        "dataset_description": data.get("dataset_description"),
                        "publication_doi": data.get("publication_doi"),
                        "publication_title": data.get("publication_title", "Title not available"),  # Add fallback
                        "publication_date": data.get("publication_date"),
                        "publication_abstract": data.get("publication_abstract", "Abstract not available")
                        # Add fallback
                    }

                    # Filter out rows missing essential IDs
                    if prepared_row["dataset_cmr_id"] and prepared_row["publication_doi"]:
                        data_chunk.append(prepared_row)
                    else:
                        total_skipped += 1

                    if len(data_chunk) >= chunk_size:
                        with driver.session() as session:
                            session.execute_write(ingest_data_transaction, data_chunk)
                        total_loaded += len(data_chunk)
                        logging.info(f"Loaded {total_loaded} relationships...")
                        data_chunk = []

                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line: {line}")
                    total_skipped += 1

        # Load any remaining data in the last chunk
        if data_chunk:
            with driver.session() as session:
                session.execute_write(ingest_data_transaction, data_chunk)
            total_loaded += len(data_chunk)
            logging.info(f"Loaded {total_loaded} relationships...")

        logging.info(f"[SUCCESS] Real data ingestion complete.")
        logging.info(f"Total relationships loaded: {total_loaded}")
        logging.info(f"Total rows skipped (missing IDs/malformed): {total_skipped}")

        # Verify by getting counts
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n) AS label, count(*) AS count")
            logging.info("--- Graph Counts ---")
            for record in result:
                if record["label"]:
                    logging.info(f"{record['label'][0]}: {record['count']} nodes")
                else:
                    logging.info(f"Nodes with no label: {record['count']}")

            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            for record in result:
                logging.info(f"{record['type']}: {record['count']} relationships")

    except Exception as e:
        logging.error(f"An error occurred during ingestion: {e}")
        logging.exception(e)  # More detailed error
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()

