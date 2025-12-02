import requests
import time
from neo4j import GraphDatabase

# 1. API CONFIGURATION
FIELD_ID = "19"
YOUR_EMAIL = "tejas.nisar@gwu.edu"
PAPERS_PER_PAGE = 200
TOTAL_PAGES_TO_FETCH = 10  # (5 * 200 = 1000 papers)

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
NEO4J_PASSWORD = ("Letmelogin@1")  # <-- UPDATE THIS
NEO4J_DATABASE = "recsys"  # <-- CONFIRMED

# 3. NEW BATCH CYPHER QUERY (Papers, Authors, and Relationships)
BATCH_CREATE_QUERY = """
UNWIND $papers_list AS paper

// 1. Merge the Paper
MERGE (p:Paper {id: paper.id})
ON CREATE SET
    p.title = paper.title,
    p.publication_year = paper.pub_year,
    p.doi = paper.doi,
    p.openalex_url = paper.id

// 2. --- THIS IS THE FIX ---
//    Pass the paper node (p) and the original paper data (paper)
//    to the next part of the query.
WITH p, paper

// 3. Loop through this paper's authors (using the 'paper' variable)
UNWIND paper.authorships AS auth_ship

// 4. Merge the Author
MERGE (a:Author {id: auth_ship.author.id})
ON CREATE SET
    a.name = auth_ship.author.display_name

// 5. Merge the WROTE Relationship (using the 'p' variable)
MERGE (a)-[r:WROTE]->(p)
ON CREATE SET
    r.author_position = auth_ship.author_position
"""

def format_papers_for_neo4j(papers_json):
    """
    Converts the JSON response into a list for Neo4j.
    This now includes the 'authorships' array.
    """
    papers_list = []
    for paper in papers_json:
        if not paper.get('id') or not paper.get('authorships'):
            continue  # Skip if no ID or no authors

        # We only need author ID, name, and position
        # We filter the 'authorships' list to simplify it
        clean_authorships = []
        for auth in paper.get('authorships', []):
            if auth.get('author') and auth['author'].get('id'):
                clean_authorships.append({
                    "author": {
                        "id": auth['author']['id'],
                        "display_name": auth['author'].get('display_name')
                    },
                    "author_position": auth.get('author_position')
                })

        papers_list.append({
            "id": paper.get('id'),
            "title": paper.get('display_name'),
            "pub_year": paper.get('publication_year'),
            "doi": paper.get('doi'),
            "authorships": clean_authorships  # <-- ADDED
        })
    return papers_list


def ingest_batch(driver, papers_list):
    """Ingests the batch of papers and authors in one transaction."""
    try:
        driver.execute_query(
            BATCH_CREATE_QUERY,  # <-- Using the new query
            papers_list=papers_list,
            database_=NEO4J_DATABASE,
        )
    except Exception as e:
        print(f"Error during batch ingestion: {e}")


# --- MAIN EXECUTION (Same as before) ---
if __name__ == "__main__":

    next_cursor = "*"
    current_page = 0

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as db_driver:
            db_driver.verify_connectivity()
            print("Successfully connected to Neo4j.")

            while current_page < TOTAL_PAGES_TO_FETCH and next_cursor:
                current_page += 1
                print(f"--- Fetching Page {current_page}/{TOTAL_PAGES_TO_FETCH} ---")

                paginated_api_url = f"{base_api_url}&cursor={next_cursor}"

                try:
                    response = requests.get(paginated_api_url)
                    response.raise_for_status()
                    data = response.json()

                    papers_json_data = data.get('results', [])

                    if papers_json_data:
                        papers_to_ingest = format_papers_for_neo4j(papers_json_data)

                        print(f"Ingesting {len(papers_to_ingest)} papers and their authors...")
                        ingest_batch(db_driver, papers_to_ingest)

                        next_cursor = data.get('meta', {}).get('next_cursor')
                        print("Ingestion for this page complete.")
                    else:
                        print("No more results found.")
                        break

                    if not next_cursor:
                        print("All pages fetched.")
                        break

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data from OpenAlex: {e}")
                    break

                time.sleep(1)

            print(f"\nTotal ingestion complete. Fetched {current_page} pages.")

    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")