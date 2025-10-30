#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for our plots
sns.set_style("whitegrid")

#%%
JSON_FILE_PATH = "data/recsys-export.json"  # Make sure this path is correct!

nodes_raw = []
relationships_raw = []

with open(JSON_FILE_PATH, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            item_type = item.get('type')
            if item_type == 'node':
                nodes_raw.append(item)
            elif item_type == 'relationship':
                relationships_raw.append(item)
        except json.JSONDecodeError as e:
            print(f"Skipping a malformed line: {e}")

print(f"Successfully loaded {len(nodes_raw)} nodes and {len(relationships_raw)} relationships.")

#%%
node_map = {node['id']: node for node in nodes_raw}  # Map of {internal_id: node_object}

papers_full_data = []
papers_stub_count = 0

for internal_id, node in node_map.items():
    properties = node.get('properties', {})
    labels = node.get('labels', [])

    if 'Paper' in labels:
        # We define a "Full Paper" as one that has a title.
        if properties.get('title'):
            papers_full_data.append(properties)
        else:
            papers_stub_count += 1

print(f"Processing complete.")
print(f"Found {len(papers_full_data)} Full Papers (with titles).")
print(f"Found {papers_stub_count} Stub Papers (from citations).")

#%%
# Create the DataFrame
papers_df = pd.DataFrame(papers_full_data)

# Convert publication_year to a numeric type, handling errors
if 'publication_year' in papers_df.columns:
    papers_df['publication_year'] = pd.to_numeric(papers_df['publication_year'], errors='coerce')

print("Papers DataFrame created.")
papers_df.head()

#%%
# Data for the plot
total_full_papers = len(papers_df)
node_types = ['Full Papers', 'Stub Papers']
node_counts = [total_full_papers, papers_stub_count]

# Create the bar plot
plt.figure(figsize=(8, 5))
barplot = sns.barplot(x=node_types, y=node_counts)

# Add titles and labels
plt.title('Count of "Full" vs. "Stub" Paper Nodes', fontsize=16)
plt.ylabel('Total Count (Log Scale)', fontsize=12)
plt.yscale('log') # Use a log scale because the difference is so massive
plt.text(0, total_full_papers, f'{total_full_papers:,}', ha='center', va='bottom', fontsize=12)
plt.text(1, papers_stub_count, f'{papers_stub_count:,}', ha='center', va='bottom', fontsize=12)

print(f"Total 'Full' Papers: {total_full_papers}")
print(f"Total 'Stub' Papers: {papers_stub_count}")

plt.show()

#%%
print("--- Schema of 'Full Papers' DataFrame ---")
papers_df.info()

# Let's get the exact percentage of null abstracts
abstract_percentage = (papers_df['abstract'].notnull().sum() / total_full_papers) * 100
print(f"\n{abstract_percentage:.2f}% of our 'Full Papers' have an abstract.")

#%%
# Filter out any papers that are missing a year or are from before 1900
valid_years_df = papers_df.dropna(subset=['publication_year'])
valid_years_df = valid_years_df[valid_years_df['publication_year'] > 1900]

print(f"Min Year: {valid_years_df['publication_year'].min():.0f}")
print(f"Max Year: {valid_years_df['publication_year'].max():.0f}")

# Create the histogram
plt.figure(figsize=(12, 6))
sns.histplot(valid_years_df['publication_year'], bins=50, kde=False)
plt.title('Distribution of Paper Publication Years (1900-Present)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.show()


#%%
# Create a list of author data from our main node_map
authors_data = []
for internal_id, node in node_map.items():
    if 'Author' in node.get('labels', []):
        authors_data.append(node.get('properties', {}))

# Create the DataFrame
authors_df = pd.DataFrame(authors_data)

print(f"Total Authors found: {len(authors_df)}")
authors_df.head()

#%%
# We'll use our node_map to link internal IDs to author names
author_id_to_name_map = {}
for internal_id, node in node_map.items():
    if 'Author' in node.get('labels', []):
        author_name = node.get('properties', {}).get('name')
        if author_name:
            # We map the *internal Neo4j ID* (e.g., '200') to the name
            author_id_to_name_map[internal_id] = author_name

# Now, count how many papers each author has
author_links = []
for rel in relationships_raw:
    if rel.get('label') == 'WROTE':
        # --- THIS IS THE FIX ---
        # The ID is inside the 'start' object
        author_internal_id = rel.get('start', {}).get('id')

        # Look up the author's name using the internal ID
        if author_internal_id in author_id_to_name_map:
            author_links.append(author_id_to_name_map[author_internal_id])

# Convert this list into a Pandas Series to easily count occurrences
author_counts = pd.Series(author_links).value_counts()

print("--- Top 10 Most Prolific Authors ---")
print(author_counts.head(10))
#%%

# Get the top 10 authors
top_10_authors = author_counts.head(10)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_authors.values, y=top_10_authors.index, palette="viridis", hue = top_10_authors.index, legend=False)

# Add titles and labels
plt.title('Top 10 Most Prolific Authors', fontsize=16)
plt.xlabel('Number of Papers Written', fontsize=12)
plt.ylabel('Author Name', fontsize=12)
plt.show()

#%%
# Create a map of {internal_id: topic_name}
topic_id_to_name_map = {}
for internal_id, node in node_map.items():
    if 'Topic' in node.get('labels', []):
        topic_name = node.get('properties', {}).get('name')
        if topic_name:
            topic_id_to_name_map[internal_id] = topic_name

# Count papers for each topic
topic_links = []
for rel in relationships_raw:
    if rel.get('label') == 'HAS_TOPIC':
        # --- THIS IS THE FIX ---
        # The ID is inside the 'end' object
        topic_internal_id = rel.get('end', {}).get('id')

        if topic_internal_id in topic_id_to_name_map:
            topic_links.append(topic_id_to_name_map[topic_internal_id])

topic_counts = pd.Series(topic_links).value_counts()

print(f"Total unique topics: {len(topic_counts)}")
print("\n--- Top 10 Most Common Topics ---")
print(topic_counts.head(10))
#%%
# Get the top 10 topics
top_10_topics = topic_counts.head(10)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_topics.values, y=top_10_topics.index, palette="mako")

# Add titles and labels
plt.title('Top 10 Most Common Topics', fontsize=16)
plt.xlabel('Number of Papers', fontsize=12)
plt.ylabel('Topic Name', fontsize=12)
plt.show()

#%%
# Create a map of {internal_id: dataset_name}
dataset_id_to_name_map = {}
for internal_id, node in node_map.items():
    if 'Dataset' in node.get('labels', []):
        dataset_name = node.get('properties', {}).get('name')
        if dataset_name:
            dataset_id_to_name_map[internal_id] = dataset_name

# Count papers for each dataset
dataset_links = []
for rel in relationships_raw:
    if rel.get('label') == 'USES_DATASET':
        # --- THIS IS THE FIX ---
        # The ID is inside the 'end' object
        dataset_internal_id = rel.get('end', {}).get('id')

        if dataset_internal_id in dataset_id_to_name_map:
            dataset_links.append(dataset_id_to_name_map[dataset_internal_id])

dataset_counts = pd.Series(dataset_links).value_counts()

print(f"Total unique datasets found: {len(dataset_counts)}")
print("\n--- Top 10 Most Used Datasets ---")
print(dataset_counts.head(10))
#%%
# --- 1. Find all paper-to-topic links ---
# We need to map the *paper's internal ID* to its topic name
paper_to_topic = {}
for rel in relationships_raw:
    if rel.get('label') == 'HAS_TOPIC':
        paper_id = rel.get('start', {}).get('id')
        topic_id = rel.get('end', {}).get('id')

        if paper_id and topic_id in topic_id_to_name_map:
            # Use .setdefault to create a list if one doesn't exist
            paper_to_topic.setdefault(paper_id, []).append(topic_id_to_name_map[topic_id])

# --- 2. Find all paper-to-dataset links ---
# We need to map the *paper's internal ID* to its dataset name
paper_to_dataset = {}
for rel in relationships_raw:
    if rel.get('label') == 'USES_DATASET':
        paper_id = rel.get('start', {}).get('id')
        dataset_id = rel.get('end', {}).get('id')

        if paper_id and dataset_id in dataset_id_to_name_map:
            paper_to_dataset.setdefault(paper_id, []).append(dataset_id_to_name_map[dataset_id])

# --- 3. Find co-occurrences ---
# Now, find all (topic, dataset) pairs that share a paper
co_occurrence_pairs = []
for paper_id, topics in paper_to_topic.items():
    # Check if this paper also has datasets
    if paper_id in paper_to_dataset:
        datasets = paper_to_dataset[paper_id]
        # Create all combinations
        for topic in topics:
            for dataset in datasets:
                co_occurrence_pairs.append((topic, dataset))

# --- 4. Count the pairs ---
co_occurrence_counts = pd.Series(co_occurrence_pairs).value_counts()

# --- 5. Prepare data for the heatmap ---
# This converts the (Topic, Dataset) index into columns
heatmap_data = co_occurrence_counts.reset_index()
heatmap_data.columns = ['pair', 'count']
heatmap_data[['Topic', 'Dataset']] = pd.DataFrame(heatmap_data['pair'].tolist(), index=heatmap_data.index)

# --- 6. Filter to only the Top N for a cleaner plot ---
# Get the names of the top 15 topics and top 15 datasets
top_15_topics = topic_counts.head(15).index
top_15_datasets = dataset_counts.head(15).index

# Filter our heatmap data to only include these top items
heatmap_data_filtered = heatmap_data[
    (heatmap_data['Topic'].isin(top_15_topics)) &
    (heatmap_data['Dataset'].isin(top_15_datasets))
    ]

# --- 7. Pivot the data into a 2D matrix ---
if not heatmap_data_filtered.empty:
    heatmap_matrix = heatmap_data_filtered.pivot(index='Topic', columns='Dataset', values='count').fillna(0)

    # --- 8. Plot the heatmap ---
    plt.figure(figsize=(18, 14))  # Make it big to be readable
    sns.heatmap(
        heatmap_matrix,
        annot=True,  # Show the numbers in the cells
        fmt=".0f",  # Format as integers
        cmap="viridis",  # Color scheme
        linewidths=.5  # Add lines between cells
    )
    plt.title('Co-occurrence of Top Topics and Datasets', fontsize=20)
    plt.xlabel('Dataset Name', fontsize=14)
    plt.ylabel('Topic Name', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()
else:
    print("No co-occurrences found between the top topics and datasets to plot.")