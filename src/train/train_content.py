import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.loader import DataLoader
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.graph_loader import Neo4jGraphLoader
from src.models.gnn import ContentSAGE
import config

# --- CONFIGURATION ---
# Use 'mps' for Mac, 'cuda' for Nvidia, else 'cpu'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
INPUT_CHANNELS = 384   # Size of BERT vectors
HIDDEN_CHANNELS = 256
OUT_CHANNELS = 128
LR = 0.001
EPOCHS = 20
BATCH_SIZE = 4096

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data (Homogeneous Graph)
    print("Loading Graph Data...")
    loader = Neo4jGraphLoader()
    data = loader.load_content_data()
    loader.close()
    
    # 2. Fix Direction (CRITICAL FIX)
    # We convert P->D into P<->D so the splitter is happy
    print("Converting to Undirected graph...")
    data = ToUndirected()(data)
    print(f"Data Ready: {data.num_nodes} nodes, {data.num_edges} edges")

    # 3. Split Data
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = transform(data)
    
    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    # 4. Initialize Model
    model = ContentSAGE(
        in_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training Loop
    print("\n--- Starting Training ---")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # A. Forward Pass
        z = model(train_data.x, train_data.edge_index)
        
        # B. Positive Samples
        pos_edge_index = train_data.edge_label_index[:, train_data.edge_label == 1]
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        
        # C. Negative Samples
        neg_edge_index = torch.randint(0, data.num_nodes, (2, pos_edge_index.size(1)), device=DEVICE)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        
        # D. Loss (Margin Loss)
        loss = (1 - pos_out + neg_out).clamp(min=0).mean()
        
        loss.backward()
        optimizer.step()
        
        # E. Validation
        val_acc = test(model, val_data)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # 6. Save Model & Embeddings
    print("\nTraining Complete!")
    os.makedirs("src/models", exist_ok=True)
    torch.save(model.state_dict(), "src/models/content_sage.pt")
    
    save_embeddings(model, data)

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        
        # Compare Positive vs Negative Scores
        pos_mask = data.edge_label == 1
        pos_idx = data.edge_label_index[:, pos_mask]
        pos_scores = (z[pos_idx[0]] * z[pos_idx[1]]).sum(dim=-1)
        
        neg_mask = data.edge_label == 0
        neg_idx = data.edge_label_index[:, neg_mask]
        neg_scores = (z[neg_idx[0]] * z[neg_idx[1]]).sum(dim=-1)
        
        # Simple Accuracy: What % of positive edges have a score > average negative score?
        threshold = neg_scores.mean()
        acc = (pos_scores > threshold).float().mean()
        return acc.item()

def save_embeddings(model, data):
    print("Generating final embeddings...")
    model.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        # Generate vectors for ALL nodes
        embeddings = model(data.x, data.edge_index).cpu().numpy()
    
    print(f"Saving {len(embeddings)} embeddings to Neo4j...")
    
    loader = Neo4jGraphLoader()
    
    # 1. Fetch IDs AND Labels so we know where to write
    query_fetch = """
    MATCH (n) 
    WHERE (n:Publication OR n:Dataset) AND n.embedding_bert IS NOT NULL
    RETURN n.globalId as id, head(labels(n)) as label
    """
    
    with loader.driver.session() as session:
        result = session.run(query_fetch)
        # Store as: {'Publication': [{'id': '...', 'vector': ...}], 'Dataset': [...]}
        updates = {'Publication': [], 'Dataset': []}
        
        for i, record in enumerate(result):
            if i < len(embeddings):
                label = record['label']
                if label in updates:
                    updates[label].append({
                        'id': record['id'], 
                        'vector': embeddings[i].tolist()
                    })
    
    # 2. Write in Batches (Split by Label to use Indexes)
    for label, batch_data in updates.items():
        if not batch_data: continue
        
        print(f"Writing {len(batch_data)} {label}s...")
        
        # Explicit MATCH using the Label (e.g., n:Publication) triggers the Index
        query_write = f"""
        UNWIND $batch as item
        MATCH (n:{label} {{globalId: item.id}}) 
        SET n.embedding_gnn_content = item.vector
        """
        
        BATCH_SIZE = 2000
        with loader.driver.session() as session:
            for i in range(0, len(batch_data), BATCH_SIZE):
                batch = batch_data[i : i + BATCH_SIZE]
                session.run(query_write, batch=batch)
                
    print("Done! Check 'embedding_gnn_content' in Neo4j.")
    loader.close()

if __name__ == "__main__":
    train()