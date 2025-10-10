# **Advanced ML Project 5: Dataset Recommendation for Researchers**

## **Project Goal**

Objective:  
Recommend Earth science datasets to researchers by comparing the performance of graph-based collaborative embeddings (network structure) versus content-driven embeddings (text descriptions).


## **✅ Completed Checklist (P1 & P2)**

* \[X\] GitHub repository setup and environment configuration.  
* \[ \] Neo4j Desktop installed and verified.  
* \[ \] Python dependencies installed (requirements.txt).  
* \[ \] config.py created and local password updated.  
* \[ \] Phase 1 Ingestion Script executed (Graph populated).  
* \[ \] Phase 2 Feature Extraction Script executed (Embeddings added).

## **🛠️ Technology Stack & Dependencies**

| Tool | Purpose in Project |
| :---- | :---- |
| **Graph Database** | Neo4j: Stores the Heterogeneous Graph (Nodes: Researcher, Paper, Dataset). |
| **Graph ML** | PyTorch Geometric (PyG): Framework for implementing Graph Neural Networks (GNNs). |
| **NLP** | sentence-transformers: Used to generate the content embeddings (vectors from abstracts). |
| **Core Language** | Python 3.11+ |

## **🚀 Setup Instructions for Teammates**



### **1\. Database Setup**

1. **Install Neo4j Desktop** and create a local database named project\_5\_recsys (default port: neo4j://localhost:7687).  
2. **Start** the database instance.  
3. **Update config.py**: Change NEO4J\_PASSWORD to your local password.

### **2\. Project Synchronization**

1. **Clone** the repository.  
2. **Create Virtual Environment** (recommended).  
3. **Install dependencies**: pip install \-r requirements.txt  
4. **Create data directory**: mkdir data

## **🧪 Phase 1 & 2 Execution (Synchronization Steps)**

Since the database is not shared, you must run the ingestion scripts locally to synchronize your graph structure with the team's current status.

### **Step 1: Initialize the Graph (P1)**

* **Action**: This script clears your local database and builds the node and relationship structure using simulated data.  
* **Command**: python ingestion\_phase\_1.py  
* **Verification**: Run MATCH () RETURN count(\*) in Neo4j Browser. Should show 12 nodes, 21 relationships.

### **Step 2: Add Features (P2)**

* **Action**: This script adds the simulated 768-dim **Content** and 128-dim **Collaborative** embeddings to all relevant nodes.  
* **Command**: python feature\_extraction\_phase\_2.py  
* **Verification**: In Neo4j Browser, run MATCH (d:Dataset) RETURN d.content\_embedding to verify the new property exists.

## **⏭️ Phase 3: Next Steps (GNN Training)**

The next step is **3.1 Data Preparation for PyTorch**. We will create the Python script necessary to pull the graph data, including the embeddings, and convert it into a PyTorch Geometric Data object, ready for model training.