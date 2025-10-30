import requests
import time
from neo4j import GraphDatabase


# --- NEW HELPER FUNCTION ---
def deinvert_abstract(inverted_index):
    """Converts an OpenAlex inverted_index back into a string."""
    if not inverted_index:
        return None
    try:
        # Find the highest index to determine the abstract's length
        max_index = -1
        for indices in inverted_index.values():
            max_index = max(max_index, max(indices))

        if max_index == -1:
            return ""  # Empty abstract

        # Create a list of empty strings
        abstract_list = [""] * (max_index + 1)

        # Populate the list with words
        for word, indices in inverted_index.items():
            for index in indices:
                abstract_list[index] = word

        # Join all words with a space
        return " ".join(abstract_list)
    except Exception as e:
        print(f"Error de-inverting abstract: {e}")
        return None


# 1. API CONFIGURATION
FIELD_ID = "19"
YOUR_EMAIL = "tejas.nisar@gwu.edu"
PAPERS_PER_PAGE = 200
TOTAL_PAGES_TO_FETCH = 20  # (1000 papers total)

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

# 3. FINAL BATCH CYPHER QUERY (with abstract)
BATCH_CREATE_QUERY = """
UNWIND $papers_list AS paper

// 1. Merge the Paper
MERGE (p:Paper {id: paper.id})
ON CREATE SET
    p.title = paper.title,
    p.publication_year = paper.pub_year,
    p.doi = paper.doi,
    p.openalex_url = paper.id,
    p.abstract = paper.abstract  // <-- NEW PROPERTY

// 2. Pass 'p' and 'paper'
WITH p, paper

// 3. Loop through authors
UNWIND paper.authorships AS auth_ship
MERGE (a:Author {id: auth_ship.author.id})
ON CREATE SET
    a.name = auth_ship.author.display_name
MERGE (a)-[r:WROTE]->(p)
ON CREATE SET
    r.author_position = auth_ship.author_position

// 4. Pass 'p' and 'paper'
WITH p, paper

// 5. Loop through referenced works
UNWIND paper.referenced_works AS cited_paper_id
MERGE (cited_p:Paper {id: cited_paper_id})
MERGE (p)-[:CITES]->(cited_p)

// 6. Pass 'p' and 'paper'
WITH p, paper

// 7. Loop through topics
UNWIND paper.topics AS topic_data
MERGE (t:Topic {id: topic_data.id})
ON CREATE SET
    t.name = topic_data.display_name
MERGE (p)-[r:HAS_TOPIC]->(t)
ON CREATE SET
    r.score = topic_data.score
"""


def format_papers_for_neo4j(papers_json):
    """
    Formats the JSON response, now including the de-inverted abstract.
    """
    papers_list = []
    for paper in papers_json:
        if not paper.get('id'):
            continue

        # --- Clean Authorships ---
        clean_authorships = []
        if paper.get('authorships'):
            for auth in paper.get('authorships', []):
                if auth.get('author') and auth['author'].get('id'):
                    clean_authorships.append({
                        "author": {
                            "id": auth['author']['id'],
                            "display_name": auth['author'].get('display_name')
                        },
                        "author_position": auth.get('author_position')
                    })

        # --- Clean Referenced Works ---
        clean_refs = []
        if paper.get('referenced_works'):
            ref_ids = set(paper.get('referenced_works', []))
            clean_refs = [ref_id for ref_id in ref_ids if ref_id]

        # --- Clean Topics ---
        clean_topics = []
        if paper.get('topics'):
            for topic in paper.get('topics', []):
                if topic.get('id') and topic.get('display_name') and topic.get('score'):
                    clean_topics.append({
                        "id": topic['id'],
                        "display_name": topic['display_name'],
                        "score": topic['score']
                    })

        # --- De-invert Abstract (NEW) ---
        raw_abstract_index = paper.get('abstract_inverted_index')
        clean_abstract = deinvert_abstract(raw_abstract_index)

        papers_list.append({
            "id": paper.get('id'),
            "title": paper.get('display_name'),
            "pub_year": paper.get('publication_year'),
            "doi": paper.get('doi'),
            "abstract": clean_abstract,  # <-- ADDED
            "authorships": clean_authorships,
            "referenced_works": clean_refs,
            "topics": clean_topics
        })
    return papers_list


def ingest_batch(driver, papers_list):
    """Ingests the batch."""
    try:
        driver.execute_query(
            BATCH_CREATE_QUERY,
            papers_list=papers_list,
            database_=NEO4J_DATABASE,
        )
    except Exception as e:
        print(f"Error during batch ingestion: {e}")


def create_indexes(driver):
    """Creates all necessary indexes."""
    print("Ensuring indexes are created...")
    try:
        driver.execute_query("CREATE INDEX paper_id_index IF NOT EXISTS FOR (p:Paper) ON (p.id)",
                             database_=NEO4J_DATABASE)
        driver.execute_query("CREATE INDEX author_id_index IF NOT EXISTS FOR (a:Author) ON (a.id)",
                             database_=NEO4J_DATABASE)
        driver.execute_query("CREATE INDEX topic_id_index IF NOT EXISTS FOR (t:Topic) ON (t.id)",
                             database_=NEO4J_DATABASE)
        print("Indexes are ready.")
    except Exception as e:
        print(f"Error creating indexes: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    next_cursor = "*"
    current_page = 0

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as db_driver:
            db_driver.verify_connectivity()
            print("Successfully connected to Neo4j.")

            create_indexes(db_driver)

            while current_page < TOTAL_PAGES_TO_FETCH:
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

                        print(f"Ingesting {len(papers_to_ingest)} papers and all related data...")
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