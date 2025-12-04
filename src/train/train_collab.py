import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.loader import LinkNeighborLoader
import sys
import os

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.graph_loader import Neo4jGraphLoader
from src.models.gnn import CollaborativeLightGCN
import config

# --- CONFIGURATION ---
# Use 'mps' for Mac, 'cuda' for Nvidia, 'cpu' otherwise
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 64
LR = 0.001
EPOCHS = 20
BATCH_SIZE = 10000  # LightGCN handles large batches well

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading Collaborative Data...")
    loader = Neo4jGraphLoader()
    data = loader.load_collaborative_data()
    
    # Retrieve mappings so we can reverse them later
    # loader.mappings is {'author': {id: idx}, 'dataset': {id: idx}}
    author_map = loader.mappings['author']
    dataset_map = loader.mappings['dataset']
    loader.close()
    
    num_authors = data['author'].num_nodes
    num_datasets = data['dataset'].num_nodes
    print(f"Graph Loaded: {num_authors} Authors, {num_datasets} Datasets.")

    # 2. Convert to Homogeneous (Unified) Graph for LightGCN
    # We map Dataset IDs to range [num_authors, num_authors + num_datasets]
    print("Converting to Unified LightGCN format...")
    
    # Get the raw edges [2, Num_Edges]
    edge_index = data['author', 'uses', 'dataset'].edge_index
    
    # Offset the destination nodes (Datasets) by the number of authors
    src = edge_index[0]
    dst = edge_index[1] + num_authors
    
    # Create unified edge index
    unified_edge_index = torch.stack([src, dst], dim=0)
    
    # Make it undirected (LightGCN needs bi-directional flow)
    # This doubles the edges: A->D becomes A<->D
    unified_edge_index = to_undirected(unified_edge_index)
    
    unified_edge_index = unified_edge_index.to(DEVICE)
    
    # 3. Initialize Model
    model = CollaborativeLightGCN(
        num_users=num_authors,
        num_items=num_datasets,
        embedding_dim=EMBEDDING_DIM,
        num_layers=3
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training Loop
    print("\n--- Starting LightGCN Training ---")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # A. Forward Pass (Propagate embeddings)
        # Returns all user and item embeddings
        user_emb, item_emb = model(unified_edge_index)
        
        # B. BPR Loss Calculation
        # We need to sample Positive edges and Negative edges
        
        # Get positive edges (Authors -> Datasets) from the original src/dst list
        # We perform random slicing for the batch
        perm = torch.randperm(src.size(0), device=DEVICE)[:BATCH_SIZE]
        batch_users = src.to(DEVICE)[perm]
        batch_pos_items = dst.to(DEVICE)[perm] - num_authors # Shift back to 0-index for lookup
        
        # Sample Negative Items (Datasets the author did NOT use)
        batch_neg_items = torch.randint(0, num_datasets, (BATCH_SIZE,), device=DEVICE)
        
        # Get the embeddings for the batch
        users = user_emb[batch_users]
        pos_items = item_emb[batch_pos_items]
        neg_items = item_emb[batch_neg_items]
        
        # Calculate Scores (Dot Product)
        pos_scores = (users * pos_items).sum(dim=-1)
        neg_scores = (users * neg_items).sum(dim=-1)
        
        # BPR Loss: -log(sigmoid(pos - neg))
        # We want pos_score > neg_score
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        # Regularization (L2 norm) prevents overfitting
        reg_loss = (1/2) * (users.norm(2).pow(2) + 
                            pos_items.norm(2).pow(2) + 
                            neg_items.norm(2).pow(2)) / float(BATCH_SIZE)
        
        total_loss = loss + (1e-4 * reg_loss)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:03d}: Loss: {total_loss.item():.4f}")

    print("\nTraining Complete!")
    os.makedirs("src/models", exist_ok=True)
    torch.save(model.state_dict(), "src/models/collab_lightgcn.pt")
    
    # 5. Save Embeddings
    save_embeddings(model, unified_edge_index, author_map, dataset_map)

def save_embeddings(model, edge_index, author_map, dataset_map):
    print("Generating final embeddings...")
    model.eval()
    with torch.no_grad():
        # Get final propagated embeddings
        user_emb, item_emb = model(edge_index)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()
    
    print(f"Saving {len(user_emb)} Authors and {len(item_emb)} Datasets to Neo4j...")
    
    loader = Neo4jGraphLoader()
    
    # Prepare Author Updates (Reverse map: Index -> OpenAlexId)
    # author_map is {openAlexId: index}
    idx_to_author = {v: k for k, v in author_map.items()}
    author_updates = []
    for i in range(len(user_emb)):
        if i in idx_to_author:
            author_updates.append({
                'id': idx_to_author[i],
                'vector': user_emb[i].tolist()
            })
            
    # Prepare Dataset Updates
    idx_to_dataset = {v: k for k, v in dataset_map.items()}
    dataset_updates = []
    for i in range(len(item_emb)):
        if i in idx_to_dataset:
            dataset_updates.append({
                'id': idx_to_dataset[i],
                'vector': item_emb[i].tolist()
            })
            
    # Write Authors
    print(f"Writing {len(author_updates)} Authors...")
    query_author = """
    UNWIND $batch as item
    MATCH (a:Author {openAlexId: item.id})
    SET a.embedding_gnn_collab = item.vector
    """
    write_in_batches(loader, query_author, author_updates)
    
    # Write Datasets
    print(f"Writing {len(dataset_updates)} Datasets...")
    query_dataset = """
    UNWIND $batch as item
    MATCH (d:Dataset {globalId: item.id})
    SET d.embedding_gnn_collab = item.vector
    """
    write_in_batches(loader, query_dataset, dataset_updates)
    
    loader.close()
    print("Done! Check 'embedding_gnn_collab' in Neo4j.")

def write_in_batches(loader, query, data):
    BATCH_SIZE = 5000
    with loader.driver.session() as session:
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            session.run(query, batch=batch)
            if i % 20000 == 0 and i > 0:
                print(f"  Written {i}...")

if __name__ == "__main__":
    train()