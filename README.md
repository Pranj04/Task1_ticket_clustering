# Ticket Clustering with NLP and Machine Learning

## Project Overview

This project clusters similar customer-support/helpdesk style tickets using semantic sentence embeddings and KMeans clustering. It includes EDA, modular preprocessing, model-based embeddings, cluster evaluation, automatic label generation, representative examples, UMAP visualization, and CSV exports.

## Dataset

The included open-source dataset is built from public GitHub issues from `pandas-dev/pandas`, stored at `data/tickets.csv`. GitHub issues are a practical support-ticket analogue because each record contains a title, description/body, metadata, and user-reported problems or requests.

You can replace this file with any CSV from Kaggle, GitHub, UCI, or another open-source helpdesk dataset. Configure the text columns in `config.json` if the dataset uses different column names.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the open-source Sentence Transformers model `all-MiniLM-L6-v2`.

## How to Run

```powershell
python main.py --input data/tickets.csv --clusters 10
```

Optional configuration:

```powershell
python main.py --input data/github_issues_tickets.csv --clusters 8 --config config.json
```

## Project Structure

```text
ticket_clustering/
|
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
└── outputs/

data/
├── github_issues_tickets.csv
└── tickets.csv

outputs/
main.py
requirements.txt
config.json
README.md
```

## Outputs

The pipeline writes:

- `outputs/eda_summary.json`: dataset shape, missing values, duplicates, and description length distribution.
- `outputs/metrics.json`: Silhouette Score and Davies-Bouldin Index.
- `outputs/clustered_tickets.csv`: final assignment output with `Ticket Description`, `Cluster ID`, and `Cluster Label`.
- `outputs/cluster_summary.csv`: cluster sizes and representative tickets.
- `outputs/cluster_visualization.png`: UMAP scatter plot of ticket clusters.

## Sample Output

```csv
Ticket Description,Cluster ID,Cluster Label
"ENH Add a way to directly access the NAType ...",2,"type / na / pylance"
"Consider adding a changelog to track version history ...",5,"documentation / changelog / release"
```

## Module Summary

- `preprocessing.py`: lowercasing, HTML removal, URL removal, special-character cleanup, whitespace normalization, and text-column preparation.
- `embedding.py`: semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- `clustering.py`: KMeans clustering, keyword-based labels, and representative tickets.
- `evaluation.py`: Silhouette Score and Davies-Bouldin Index.
- `visualization.py`: UMAP projection and matplotlib plotting.
- `utils.py`: config loading, pandas CSV loading, EDA, JSON output, and logging setup.
