"""
Script to execute a Cypher file against Neo4j database.
This loads and runs the graph.cypher file to populate the database.
"""

import logging
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logging.info("✓ Neo4j connection successful.")
        return driver
    except Exception as e:
        logging.error(f"✗ Failed to connect to Neo4j: {e}")
        return None


def execute_cypher_file(driver, filepath):
    """
    Reads a .cypher file and executes all statements.
    Handles multi-line statements separated by semicolons.
    """
    logging.info(f"Reading Cypher file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cypher_content = f.read()
        
        # Split by semicolons to handle multiple statements
        # Note: This is a simple split - won't handle semicolons inside strings properly
        statements = [stmt.strip() for stmt in cypher_content.split(';') if stmt.strip()]
        
        logging.info(f"Found {len(statements)} Cypher statements to execute.")
        
        with driver.session(database="recsys") as session:
            for i, statement in enumerate(statements, 1):
                try:
                    logging.info(f"Executing statement {i}/{len(statements)}...")
                    result = session.run(statement)
                    
                    # Try to get summary info
                    summary = result.consume()
                    
                    nodes_created = summary.counters.nodes_created
                    rels_created = summary.counters.relationships_created
                    props_set = summary.counters.properties_set
                    
                    if nodes_created > 0 or rels_created > 0 or props_set > 0:
                        logging.info(
                            f"  → Nodes created: {nodes_created}, "
                            f"Relationships created: {rels_created}, "
                            f"Properties set: {props_set}"
                        )
                    
                except Exception as e:
                    logging.error(f"✗ Error executing statement {i}: {e}")
                    logging.error(f"Statement preview: {statement[:200]}...")
                    # Continue with next statement instead of stopping
                    continue
        
        logging.info("✓ Cypher file execution complete!")
        return True
        
    except FileNotFoundError:
        logging.error(f"✗ File not found: {filepath}")
        return False
    except Exception as e:
        logging.error(f"✗ Error reading or executing file: {e}")
        return False


def verify_graph(driver):
    """Prints summary statistics of the graph."""
    logging.info("\n--- Graph Statistics ---")
    
    with driver.session(database="recsys") as session:
        # Count nodes by label
        result = session.run("MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY count DESC")
        logging.info("\nNode counts:")
        for record in result:
            if record["label"]:
                logging.info(f"  {record['label'][0]}: {record['count']:,}")
        
        # Count relationships by type
        result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC")
        logging.info("\nRelationship counts:")
        for record in result:
            logging.info(f"  {record['type']}: {record['count']:,}")
        
        # Total counts
        total_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        
        logging.info(f"\nTotal: {total_nodes:,} nodes, {total_rels:,} relationships")


def main():
    logging.info("Starting Cypher file execution script...")
    
    # Connect to Neo4j
    driver = get_neo4j_driver()
    if not driver:
        logging.error("Cannot proceed without database connection.")
        return
    
    try:
        # Execute the cypher file
        cypher_file_path = "data/graph.cypher"
        success = execute_cypher_file(driver, cypher_file_path)
        
        if success:
            # Show statistics
            verify_graph(driver)
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()
