import requests
import time
from neo4j import GraphDatabase

# 1. API CONFIGURATION
FIELD_ID = "19"
YOUR_EMAIL = "tejas.nisar@gwu.edu"
PAPERS_PER_PAGE = 200
TOTAL_PAGES_TO_FETCH = 10  # <-- We will fetch 5 pages (5 * 200 = 1000 papers)

# This is our starting URL
base_api_url = (
    f"https://api.openalex.org/works"
    f"?filter=topics.field.id:{FIELD_ID}"
    f"&sort=cited_by_count:desc"
    f"&per-page={PAPERS_PER_PAGE}"
    f"&mailto={YOUR_EMAIL}"
)

# 2. NEO4J CONFIGURATION
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Letmelogin@1"  # <-- UPDATE THIS
NEO4J_DATABASE = "recsys"

# 3. BATCH CYPHER QUERY (same as before)
BATCH_CREATE_PAPERS_QUERY = """
UNWIND $papers_list AS paper
MERGE (p:Paper {id: paper.id})
ON CREATE SET
    p.title = paper.title,
    p.publication_year = paper.pub_year,
    p.doi = paper.doi,
    p.openalex_url = paper.id
"""


def format_papers_for_neo4j(papers_json):
    """Converts the JSON response into a flat list for Neo4j."""
    papers_list = []
    for paper in papers_json:
        if not paper.get('id'):
            continue
        papers_list.append({
            "id": paper.get('id'),
            "title": paper.get('display_name'),
            "pub_year": paper.get('publication_year'),
            "doi": paper.get('doi'),
        })
    return papers_list


def ingest_papers_batch(driver, papers_list):
    """Ingests a list of papers in a single transaction."""
    try:
        driver.execute_query(
            BATCH_CREATE_PAPERS_QUERY,
            papers_list=papers_list,
            database_=NEO4J_DATABASE,
        )
    except Exception as e:
        print(f"Error during batch ingestion: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # We start with the base_api_url and add the cursor=*.
    # In the loop, this will be replaced by the 'next_cursor' value.
    next_cursor = "*"
    current_page = 0

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as db_driver:
            db_driver.verify_connectivity()
            print("Successfully connected to Neo4j.")

            while current_page < TOTAL_PAGES_TO_FETCH and next_cursor:
                current_page += 1
                print(f"--- Fetching Page {current_page}/{TOTAL_PAGES_TO_FETCH} ---")

                # Construct the full URL with the current cursor
                paginated_api_url = f"{base_api_url}&cursor={next_cursor}"

                try:
                    response = requests.get(paginated_api_url)
                    response.raise_for_status()
                    data = response.json()

                    # Get the list of papers from this page
                    papers_json_data = data.get('results', [])

                    if papers_json_data:
                        # Format data for Neo4j
                        papers_to_ingest = format_papers_for_neo4j(papers_json_data)

                        # Ingest this batch
                        print(f"Ingesting {len(papers_to_ingest)} papers from this page...")
                        ingest_papers_batch(db_driver, papers_to_ingest)

                        # Get the 'bookmark' for the next page
                        next_cursor = data.get('meta', {}).get('next_cursor')
                        print("Ingestion for this page complete.")
                    else:
                        print("No more results found.")
                        break  # Exit the loop if there are no results

                    if not next_cursor:
                        print("All pages fetched.")
                        break  # Exit the loop if the API says there's no next cursor

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data from OpenAlex: {e}")
                    break  # Stop if there's an API error

                # Be polite to the API and wait a moment before the next request
                time.sleep(1)

            print(f"\nTotal ingestion complete. Fetched {current_page} pages.")

    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")