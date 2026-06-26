# Topic Evolution Tracker & Semantic Search Engine

This repository contains a dual-module pipeline designed to analyze, track, and search textual topics across consecutive, time-series data drops. It identifies how topics shift, split, or merge over time using structural overlap metrics, filters out duplicate historical themes, and provides a semantic vector search interface to query distinct topics.

---

## Architecture Overview

The system is split into two distinct sub-systems:

1. **Topic Evolution Pipeline (`part-1.txt`)**:
* Tracks longitudinal changes across multi-day text mining runs.


* Evaluates post-level overlaps via Jaccard metrics and lexical summaries via TF-IDF cosine similarity.


* Generates automated CSV exports mapping both aggregate topic changes and post-level lineage.




2. **Deduplication & Semantic Retrieval (`part-2.txt`)**:
* Uses a token-based Jaccard similarity threshold to clear redundant topic metadata across chronological periods.


* Leverages the `multi-qa-MiniLM-L6-cos-v1` transformer model to generate dense vector embeddings of unique topics.


* Implements a real-time command-line interface for semantic queries using vector similarity matrix calculations.





---

## Features

* 
**Longitudinal Tracking**: Computes intersection and union profiles of post groupings to trace historical lineage.


* 
**Dynamic Status Classification**: Labels state variations into actionable categories based on quantitative thresholds: `Stable`, `Evolved`, `Fragmented`, or `Dropped/Noise`.


* 
**Entity Delta Monitoring**: Computes exact additions and removals of structural keywords/entities within changing topics over time.


* 
**Automated Data Cleanse**: Dedupes recurring historical concepts (Jaccard similarity threshold $> 0.5$) before indexing to optimize memory usage and vector search speed.


* 
**Deep Semantic Querying**: Translates simple text searches into sentence-level semantic representations for context-aware metadata matching.



---

## Repository Structure

```text
├── Data/
│   ├── 2026-02-19_posts.json      # Run 1 initial data drop
│   ├── 2026-02-20_posts.json      # Run 2 sequential data drop
│   └── 2026-02-21_posts.json      # Run 3 sequential data drop
├── topic_evolution.py             # Topic transition and alignment engine (Part 1)
├── semantic_search.py             # Deduplication and embedding query engine (Part 2)
├── Topic_Comparison.csv           # Generated macro topic tracking matrix
└── Post_Comparison.csv            # Generated micro post lineage timeline

```

---

## Tech Stack

* 
**Core Libraries**: `pandas`, `numpy` 


* 
**Machine Learning / NLP**: `scikit-learn` 


* 
**Deep Learning Embeddings**: `sentence-transformers` 



---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/topic-evolution-search.git
cd topic-evolution-search

```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn sentence-transformers

```

### 3. Data Requirements

Ensure your input JSON files are located in a `/Data` subdirectory relative to the script execution layout. The input documents must follow this internal schema structure:

```json
{
  "posts": [
    { "id": 1001, "topic": 1 }
  ],
  "topic_entities": [
    {
      "topic": 1,
      "title": "Example Topic Title",
      "summary": ["An array of sentence strings representing the topic summary."],
      "key_entities": [["Entity Name", "Weight/Type"]]
    }
  ]
}

```

---

## Usage

### Run Topic Evolution Analysis

Executes cross-run tracking computations and logs outputs directly to filesystem CSV matrices:

```bash
python topic_evolution.py

```

* Generates `Topic_Comparison.csv` containing macro statistics (Jaccard score, cosine similarities, entity deltas) across execution blocks.


* Generates `Post_Comparison.csv` containing post-level lineage tracking over all observed intervals.



### Launch Semantic Search Interface

Initializes the system deduplication routine, registers remaining unique clusters into vector-space coordinates, and launches an interactive search shell:

```bash
python semantic_search.py

```

**Example Interactive Workflow:**

```text
Checking for Duplicate topics...
Embedding the topics...
==================================================


Write your query: cloud infrastructure automation
Topic: 3
Title: Infrastructure as Code Best Practices
Similarity Score: 0.8142

```
