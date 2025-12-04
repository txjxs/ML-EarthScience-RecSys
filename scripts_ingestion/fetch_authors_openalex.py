import os
import time
import requests
from neo4j import GraphDatabase
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- CONFIGURATION ---
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD
OPENALEX_EMAIL = config.OPENALEX_POLITE_EMAIL

# Batch size for OpenAlex API (Max is roughly 50-100 per call for filters)
BATCH_SIZE = 100

class OpenAlexEnricher:
    def __init__(self, uri, user, password, email):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.email = email
        self.base_url = "https://api.openalex.org/works"

    def close(self):
        self.driver.close()

    def get_target_publications(self):
        """
        Find publications that have a DOI but likely missing/messy authors.
        We return a list of dicts: [{'id': globalId, 'doi': '10.xxxx/...'}]
        """
        query = """
        MATCH (p:Publication)
        WHERE p.doi IS NOT NULL 
          AND NOT (p)<-[:WROTE]-(:Author)
        RETURN p.globalId AS id, p.doi AS doi
        LIMIT 50000  // Process in chunks if you have 110k
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def fetch_from_openalex(self, dois):
        """
        Queries OpenAlex for a batch of DOIs using the pipe operator |
        Returns a dict: { doi_key: [ {id, name, orcid} ] }
        """
        # Clean DOIs (strip https://doi.org/ if present to get raw ID)
        clean_dois = [d.replace("https://doi.org/", "").strip() for d in dois]
        
        # OpenAlex filter format: doi:10.1126/science.123|10.1038/nature.456
        doi_filter = "|".join(clean_dois)
        params = {
            "filter": f"doi:{doi_filter}",
            "per_page": 200,
            "mailto": self.email
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                return {}
            
            data = resp.json()
            results = {}
            
            for work in data.get('results', []):
                work_doi = work.get('doi') # Returns full https://doi.org/10.xxx
                if not work_doi: continue
                
                # Extract clean DOI to match our key
                key_doi = work_doi.replace("https://doi.org/", "").lower()
                
                authors = []
                for authorship in work.get('authorships', []):
                    author = authorship.get('author', {})
                    if author.get('id'):
                        authors.append({
                            'openAlexId': author['id'], # e.g., https://openalex.org/A5003...
                            'name': author['display_name'],
                            'orcid': author.get('orcid')
                        })
                results[key_doi] = authors
                
            return results

        except Exception as e:
            print(f"Request Failed: {e}")
            return {}

    def update_graph(self, doi_author_map):
        """
        Writes the authors to Neo4j.
        doi_author_map format: { '10.1002/xyz': [ {id, name}... ] }
        """
        query = """
        UNWIND $batch AS item
        MATCH (p:Publication) WHERE toLower(p.doi) CONTAINS item.doi
        
        WITH p, item
        UNWIND item.authors AS auth
        
        MERGE (a:Author {openAlexId: auth.openAlexId})
        ON CREATE SET a.name = auth.name, a.orcid = auth.orcid
        
        MERGE (a)-[:WROTE]->(p)
        """
        
        # Reshape data for UNWIND
        batch_data = []
        for doi, authors in doi_author_map.items():
            if authors:
                batch_data.append({'doi': doi, 'authors': authors})
        
        if not batch_data: return

        with self.driver.session() as session:
            session.run(query, batch=batch_data)
            print(f"Updated {len(batch_data)} publications with authors.")

    def run(self):
        print("Fetching target publications from Neo4j...")
        pubs = self.get_target_publications()
        print(f"Found {len(pubs)} publications to process.")
        
        # Process in batches
        for i in range(0, len(pubs), BATCH_SIZE):
            batch = pubs[i : i + BATCH_SIZE]
            dois = [p['doi'] for p in batch]
            
            print(f"Processing batch {i} - {i + len(batch)}...")
            
            # 1. Fetch from OpenAlex
            author_data = self.fetch_from_openalex(dois)
            
            # 2. Match results back to input DOIs (handle case sensitivity)
            # We map the clean DOI from OpenAlex back to the data we need to write
            # (Logic handled inside update_graph via CONTAINS matching)
            
            # 3. Write to Neo4j
            self.update_graph(author_data)
            
            # Be polite to the API
            time.sleep(0.5)

if __name__ == "__main__":
    enricher = OpenAlexEnricher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENALEX_EMAIL)
    try:
        enricher.run()
    finally:
        enricher.close()