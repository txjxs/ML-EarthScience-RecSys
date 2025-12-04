import torch
from torch_geometric.data import HeteroData
from neo4j import GraphDatabase
import numpy as np

class Neo4jContentLoader:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def fetch_node_features(self, label, embedding_property):
        """Fetches elementId and Embedding vector for a specific node label."""
        # CHANGED: We now use elementId(n) instead of n.id
        query = f"""
        MATCH (n:{label})
        WHERE n.{embedding_property} IS NOT NULL
        RETURN elementId(n) AS id, n.{embedding_property} AS embedding
        """
        with self.driver.session(database=self.database) as session:
            results = session.run(query).data()
            
        if not results:
            print(f"WARNING: No nodes found for label '{label}' with property '{embedding_property}'")
            return torch.empty((0, 128)), {}

        # Create a mapping from Neo4j elementId -> Consecutive Integer ID (0, 1, 2...)
        id_map = {row['id']: i for i, row in enumerate(results)}
        
        # Convert embeddings to a Tensor
        embeddings = [row['embedding'] for row in results]
        x = torch.tensor(embeddings, dtype=torch.float)
        
        return x, id_map

    def fetch_edges(self, source_label, target_label, rel_type, source_map, target_map):
        """Fetches edge indices for a relationship using elementIds."""
        
        # CHANGED: We use elementId(s) and elementId(t)
        query = f"""
        MATCH (s:{source_label})-[r:{rel_type}]->(t:{target_label})
        WHERE elementId(s) IN $source_ids AND elementId(t) IN $target_ids
        RETURN elementId(s) AS source, elementId(t) AS target
        """
        
        params = {
            'source_ids': list(source_map.keys()),
            'target_ids': list(target_map.keys())
        }

        with self.driver.session(database=self.database) as session:
            results = session.run(query, params).data()

        edge_indices = []
        for row in results:
            if row['source'] in source_map and row['target'] in target_map:
                s_idx = source_map[row['source']]
                t_idx = target_map[row['target']]
                edge_indices.append([s_idx, t_idx])

        if not edge_indices:
            print(f"WARNING: No edges found for {source_label}-[:{rel_type}]->{target_label}")
            return torch.empty((2, 0), dtype=torch.long)

        # PyG expects shape [2, num_edges]
        return torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    def load_content_graph(self):
        data = HeteroData()
        print("Fetching Node Features...")

        # 1. Load Nodes & Features
        paper_x, paper_map = self.fetch_node_features("Paper", "content_embedding")
        topic_x, topic_map = self.fetch_node_features("Topic", "content_embedding")
        dataset_x, dataset_map = self.fetch_node_features("Dataset", "content_embedding")

        data['paper'].x = paper_x
        data['topic'].x = topic_x
        data['dataset'].x = dataset_x
        
        # Save maps for later lookup
        self.maps = {
            'paper': paper_map,
            'topic': topic_map,
            'dataset': dataset_map
        }

        print(f"Nodes Loaded: {len(paper_x)} Papers, {len(topic_x)} Topics, {len(dataset_x)} Datasets")

        # 2. Load Edges
        print("Fetching Edges...")
        
        # Paper <-> Topic
        edge_index_p_t = self.fetch_edges("Paper", "Topic", "HAS_TOPIC", paper_map, topic_map)
        data['paper', 'has_topic', 'topic'].edge_index = edge_index_p_t
        data['topic', 'rev_has_topic', 'paper'].edge_index = edge_index_p_t.flip(0)

        # Paper <-> Dataset
        edge_index_p_d = self.fetch_edges("Paper", "Dataset", "USES_DATASET", paper_map, dataset_map)
        data['paper', 'uses_dataset', 'dataset'].edge_index = edge_index_p_d
        data['dataset', 'rev_uses_dataset', 'paper'].edge_index = edge_index_p_d.flip(0)

        print("Graph Construction Complete.")
        return data

    def close(self):
        self.driver.close()