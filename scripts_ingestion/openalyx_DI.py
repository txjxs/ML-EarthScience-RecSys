import config
import logging
from neo4j import GraphDatabase
import pyalex # Correct import
from pyalex import Works # We only need Works

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. HELPER FUNCTIONS ---

def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    try:
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD) # Use variable from config.py
        )
        driver.verify_connectivity()
        logging.info("Neo4j connection successful.")
        return driver
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        return None

def clear_database(driver):
    """WIPES all nodes and relationships from the database."""
    logging.warning("WIPING all data from the Neo4j database...")
    try:
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        logging.info("Database successfully cleared.")
    except Exception as e:
        logging.error(f"Failed to clear database: {e}")
        raise

# --- 2. OPENALEX API FUNCTION (USING THE CORRECT FIELD ID AND SYNTAX) ---

def fetch_openalex_data(email):
    """
    Fetches papers tagged with the 'Earth science' FIELD ID (F19)
    using the correct dictionary filter syntax.
    """
    logging.info("Fetching data from OpenAlex API using 'pyalex'...")

    pyalex.config.email = email

    try:
        # *** The correct, verified Field ID from your Google Sheet. ***
        EARTH_SCIENCE_FIELD_ID = "https://openalex.org/F19"

        logging.info(f"Using verified Field ID for 'Earth science': {EARTH_SCIENCE_FIELD_ID}")

        # Filter works using the correct dictionary syntax for nested field
        logging.info("Fetching 100 works filtered by Field ID...")

        # *** The correct pyalex syntax using the dictionary filter. ***
        results = Works() \
                  .filter(topics={'field': {'id': EARTH_SCIENCE_FIELD_ID}}) \
                  .get(per_page=100) # Get 100 papers

        # Check if results is a Paginator object and get meta from it
        if hasattr(results, 'meta'):
            meta = results.meta
        else:
            # Handle cases where the result might be just the list (older pyalex versions?)
            meta = getattr(results, 'meta', {}) # Safely get meta if available

        total_count = meta.get('count', 'unknown')

        # 'results' from .get() without return_meta=True should be the list directly
        actual_results_list = results.results if hasattr(results, 'results') else results


        logging.info(f"Successfully fetched {len(actual_results_list)} works from OpenAlex (Total found in this field: {total_count}).")

        if len(actual_results_list) == 0:
             logging.warning("Fetched 0 works. This is unexpected for this field.")
             return None # Stop if we get nothing

        return actual_results_list # Return the list of work dictionaries

    except Exception as e:
        logging.error(f"Failed to fetch data using pyalex: {e}")
        logging.exception(e)
        return None


# --- 3. NEO4J INGESTION --- (No changes)

def ingest_papers_transaction(tx, papers_data):
    """
    The Neo4j transaction function to load all papers, authors,
    and citations in one efficient query.
    """
    query = """
    UNWIND $papers AS paper

    MERGE (p:Paper {id: paper.id})
    ON CREATE SET
        p.title = paper.title,
        p.doi = paper.doi,
        p.abstract = paper.abstract,
        p.publication_year = paper.publication_year

    FOREACH (author_data IN paper.authors |
        MERGE (a:Researcher {id: author_data.id})
        ON CREATE SET
            a.name = author_data.display_name
        MERGE (a)-[:AUTHORED]->(p)
    )

    FOREACH (cited_paper_id IN paper.referenced_works |
        MERGE (p_cited:Paper {id: cited_paper_id})
        MERGE (p)-[:CITES]->(p_cited)
    )
    """
    try:
        tx.run(query, papers=papers_data)
    except Exception as e:
        logging.error(f"Error during transaction: {e}")
        raise

# --- 4. MAIN EXECUTION --- (Small change to handle list return)

def main():
    driver = None
    try:
        driver = get_neo4j_driver()
        if not driver:
            logging.error("Halting: Neo4j driver not available.")
            return

        clear_database(driver)

        works_list = fetch_openalex_data(config.OPENALEX_POLITE_EMAIL) # Now returns a list
        if not works_list:
            logging.error("Halting: No data fetched from OpenAlex.")
            return

        papers_data_for_ingestion = []
        for work in works_list: # Iterate through the list
            authors = []
            if work.get("authorships"):
                for authorship in work["authorships"]:
                    author_info = authorship.get("author")
                    if author_info and author_info.get("id"):
                        authors.append({
                            "id": author_info["id"],
                            "display_name": author_info.get("display_name", "Name not available")
                        })

            papers_data_for_ingestion.append({
                "id": work.get("id"),
                "title": work.get("title", "Title not available"),
                "doi": work.get("doi"),
                "publication_year": work.get("publication_year"),
                "abstract": work.get("abstract", reconstruct_abstract(work.get("abstract_inverted_index"))),
                "authors": authors,
                "referenced_works": work.get("referenced_works", [])
            })

        logging.info(f"Ingesting {len(papers_data_for_ingestion)} seed papers...")
        with driver.session() as session:
            session.execute_write(ingest_papers_transaction, papers_data_for_ingestion)

        logging.info("[SUCCESS] OpenAlex data ingestion complete.")

        with driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n) AS label, count(*) AS count")
            logging.info("--- Graph Counts ---")
            for record in result:
                if record["label"]:
                    logging.info(f"{record['label'][0]}: {record['count']} nodes")

            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            for record in result:
                logging.info(f"{record['type']}: {record['count']} relationships")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        logging.exception(e)
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")

# Helper function (No changes)
def reconstruct_abstract(inv_index):
    if not inv_index: return "Abstract not available"
    try:
        word_pos = {}
        for word, positions in inv_index.items():
            for pos in positions:
                word_pos[pos] = word
        return " ".join(word_pos[i] for i in sorted(word_pos.keys()))
    except Exception as e:
        logging.warning(f"Could not parse abstract: {e}")
        return "Abstract parsing error"


if __name__ == "__main__":
    main()