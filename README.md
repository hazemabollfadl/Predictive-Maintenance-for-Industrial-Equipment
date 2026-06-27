# Predictive Maintenance for Industrial Equipment

Remaining Useful Life (RUL) prediction for aircraft engines using the [C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The project is structured for MLOps with separate pipelines, configuration files, and exploratory notebooks.

## Project Structure

```
Predictive-Maintenance-for-Industrial-Equipment/
│
├── src/
│   ├── configs/                        # Pipeline configuration (YAML)
│   │   ├── ingestion.yaml
│   │   ├── training.yaml
│   │   └── model.yaml
│   │
│   ├── data/
│   │   ├── raw/                        # Immutable source data
│   │   │   ├── training_data/          # train_FD001-4.txt
│   │   │   ├── test_data/              # test_FD001-4.txt
│   │   │   └── rul_data/              # RUL_FD001-4.txt
│   │   ├── processed/                  # Cleaned & transformed outputs
│   │   └── features/                  # Feature-engineered datasets
│   │
│   ├── scripts/                      # Exploratory analysis (not production code)
│   │   ├── data_pipeline.py
│   │   ├── main_evaluation.py
│   │   ├── model_cnn.py
│   │   ├── visuals_presentation.ipynb
│   │   └── random_forest.ipynb
│   │
│   ├── pipelines/                      # Runnable pipeline steps
│   │   ├── ingestion/
│   │   ├── preprocessing/
│   │   ├── training/
│   │   └── evaluation/
│   │
│   ├── src/                           # Shared library code (importable, no side effects)
│   │   ├── features/                  # Feature engineering functions
│   │   ├── models/                    # Model definitions
│   │   └── utils/
│   │
│   ├── models/                        # Serialized model artifacts
│   │
│   └── tests/
│       ├── unit/
│       └── integration/
│
├── references/
│   ├── architecture_notes/            # Per-phase design notes (Phases 1-4)
│   └── Damage Propagation Modeling.pdf
│                  
├── .env                               # Local environment variables (not committed)
├── .env.example                       # Environment variable template
└── requirements.txt
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (use `.env.example` as a template):

```text
DATASET_BASE_PATH = "/path/to/CMAPSSData"
```

Example on Windows:

```text
DATASET_BASE_PATH = "C:\Users\<your-username>\...\CMAPSSData"
```
