# import requests
# import pprint
#
# # This URL asks OpenAlex for all concepts that are at 'level:0' (the top level)
# api_url = "https://api.openalex.org/concepts?filter=level:0"
#
# try:
#     response = requests.get(api_url)
#     response.raise_for_status()  # This will raise an error if the request fails
#
#     data = response.json()
#
#     # Let's print the display_name and id for each concept
#     for concept in data['results']:
#         print(f"Name: {concept['display_name']}, ID: {concept['id']}")
#
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data: {e}")
#
# import requests
# import pprint
#
# # This time we search the /topics endpoint
# search_query = "Geology"
# api_url = f"https://api.openalex.org/topics?search={search_query}"
#
# try:
#     response = requests.get(api_url)
#     response.raise_for_status()
#
#     data = response.json()
#
#     # Let's look at the first few results
#     # We want to find the 'field' object inside the topic
#     if data['results']:
#         print("Found the following topics:")
#         for topic in data['results'][:5]:  # Just print the top 5
#             print(f"\nName: {topic['display_name']}, ID: {topic['id']}")
#             print(f"  -> Field: {topic.get('field')}")  # This is what we want!
#     else:
#         print("No topics found for that query.")
#
# except requests.exceptions.RequestException as e:
#     print(f"Error fetching data: {e}")
#%%
import requests
import pprint

# The Field ID for "Earth and Planetary Sciences" that you found
FIELD_ID = "19"

# Your email for the OpenAlex 'mailto' polite pool
YOUR_EMAIL = "tejas.nisar@gwu.edu"  # Please change this

# This URL fetches 'works' (papers) and filters by the Field ID
# We also sort by 'cited_by_count:desc' to get prominent papers
api_url = (
    f"https://api.openalex.org/works"
    f"?filter=topics.field.id:{FIELD_ID}"
    f"&sort=cited_by_count:desc"
    f"&per-page=10"
    f"&mailto={YOUR_EMAIL}"
)
print(api_url)
try:
    response = requests.get(api_url)
    response.raise_for_status()

    data = response.json()

    # Let's print the titles and OpenAlex IDs to confirm
    print("Found the following 'Earth and Planetary Sciences' papers:")
    for work in data['results']:
        print(f"- ID: {work.get('id')}, Title: {work.get('display_name')}")

except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")

#%%
from neo4j import GraphDatabase

# --- Update these details ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Letmelogin@1"

NEO4J_DATABASE = "ML-EarthScience-RecSys" # Default database
# ----------------------------

try:
    # Try to create a driver instance
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        # Verify that the connection is valid and the server is available
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")

except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
