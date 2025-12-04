import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LGConv



class ContentSAGE(torch.nn.Module):
    """
    Model 2: Content-Driven
    Uses GraphSAGE to learn from Text Features (BERT) + Citation Structure.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Layer 1: Aggregates immediate neighbors (384 -> 256)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # Layer 2: Aggregates neighbors of neighbors (256 -> 128)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x is the [Num_Nodes, 384] BERT matrix
        
        # 1. First Hop
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 2. Second Hop
        x = self.conv2(x, edge_index)
        
        # 3. L2 Normalization (Crucial for Cosine Similarity later)
        x = F.normalize(x, p=2, dim=1)
        return x

class CollaborativeLightGCN(torch.nn.Module):
    """
    Model 1: Collaborative Filtering
    Uses LightGCN (State-of-the-Art for RecSys).
    It simplifies GCN by removing non-linearities, focusing purely on graph propagation.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        
        # 1. Initial Embeddings (Learnable weights for every User and Item)
        # We start with random noise, unlike SAGE which starts with BERT.
        self.embedding = torch.nn.Embedding(num_users + num_items, embedding_dim)
        
        # 2. Light Graph Convolution
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(num_layers)])
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        # LightGCN works on a unified graph (Users + Items)
        # edge_index must map [User_ID] -> [Item_ID + Num_Users]
        
        # 1. Get initial 0-layer embeddings
        emb = self.embedding.weight
        embs = [emb]
        
        # 2. Propagate through layers
        for conv in self.convs:
            emb = conv(emb, edge_index)
            embs.append(emb)
            
        # 3. Average all layers (The "LightGCN" trick)
        # This captures both immediate interests and multi-hop community interests
        out = torch.stack(embs, dim=1)
        out = torch.mean(out, dim=1)
        
        # 4. Split back into Users and Items
        user_emb, item_emb = torch.split(out, [self.num_users, self.num_items])
        
        return user_emb, item_emb