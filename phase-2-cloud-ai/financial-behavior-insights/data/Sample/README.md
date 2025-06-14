# Data Folder & Versioning Guide

## What Goes Here?

- **Raw and processed data files** for the project. Do NOT commit real data (only DVC pointer files: `.dvc`).
- **Schema documentation** and data provenance notes.

## Data Versioning Workflow

- **Local experiments:**  
  All data files are tracked with DVC (`dvc add data/transactions.csv`).  
  This lets you version, roll back, and share data files without clogging git.

- **Sharing with team/CI:**  
  Set up a DVC remote (e.g., Azure Blob) for distributed access (`dvc push`/`dvc pull`).

- **Cloud pipeline use:**  
  Register each major version in Azure ML as a Data Asset for pipelines, tracking, and reproducibility.

## Data Schema (example)

| Column        | Type    | Description               |
|---------------|---------|---------------------------|
| step          | int     | Time step of transaction  |
| type          | str     | Transaction type (CASH, ...) |
| amount        | float   | Transaction amount        |
| ...           | ...     | ...                       |

## Sample Workflow

```bash
# Add new data to DVC
dvc add data/transactions.csv
git add data/transactions.csv.dvc
git commit -m "Track transactions.csv with DVC"

# Push to DVC remote (if set up)
dvc push

# Register as Azure ML Data Asset (run registration script)
python src/data/register_data_asset.py