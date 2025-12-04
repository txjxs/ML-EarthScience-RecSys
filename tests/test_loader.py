import sys
import os

# Add the src directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.neo4j_loader import Neo4jContentLoader

# CONFIGURATION
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

URI = config.NEO4J_URI
USER = config.NEO4J_USER
PASSWORD = config.NEO4J_PASSWORD
DATABASE = 'neo4j' #config.DATABASE

def test_loading():
    loader = Neo4jContentLoader(URI, USER, PASSWORD, DATABASE)
    
    try:
        data = loader.load_content_graph()
        print("\n--- Data Inspection ---")
        print(data)
        
        # Basic Checks
        assert data['paper'].x.shape[1] == 128, "Paper embeddings should be size 128"
        assert data['dataset'].x.shape[1] == 128, "Dataset embeddings should be size 128"
        
        print("\n✅ SUCCESS: Data loaded correctly into PyG format.")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    test_loading()