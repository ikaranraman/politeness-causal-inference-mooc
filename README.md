# Quantifying Politeness in Online Educational Forums
### A Computational Study of Instructor and Student Communication

**Minor Thesis | FIT5128 | Monash University**  
**Author:** Karan Raman (35134070)  
**Supervisor:** Dr. Guanliang Chen  
**Co-Supervisor:** Kaixun Yang  

---

## Overview

This project applies Natural Language Processing and causal inference to quantify politeness in 28,266 forum posts from the Stanford MOOC Forum Post dataset. Drawing on Brown and Levinson's (1987) politeness theory, each post is assigned a scalar score derived from 21 binary linguistic features extracted using ConvoKit. Causal inference methods (back-door criterion and X-learner) are then used to estimate the causal effect of five treatment variables on politeness: anonymity, question post type, high urgency, high confusion and late response timing. A feasibility study of LLM-based politeness rewriting is included as an optional extension.

---

## Research Questions

1. To what extent do posts display politeness in online MOOC forum communication?
2. How do politeness levels differ across post types, course disciplines and contextual factors?
3. What is the causal effect of anonymity, post type, urgency, confusion and response timing on politeness?
4. Can a language model improve the politeness of forum posts while preserving meaning?

---

## Key Results

- 56.8% of posts are polite, 26.4% neutral, 16.9% impolite (mean score = 0.797, SD = 1.355)
- Question posts consistently reduce politeness across all disciplines (mean ITE = -0.329)
- Urgency increases politeness strongly in Medicine (ITE = +0.214) but near zero in Humanities (ITE = +0.025)
- Placebo refutation p-values ranged 0.78 to 0.94, confirming causal estimates are not spurious
- LLM intervention: mean improvement = 0.460, mean cosine similarity = 0.815 (n=100)

---

## Pipeline

| Notebook | Purpose | Input | Output |
|---|---|---|---|
| `01_eda.ipynb` | Exploratory data analysis | `data/raw/stanfordMOOCForumPostsSet.xlsx` | `data/processed/posts_eda.parquet` |
| `02_preprocessing.ipynb` | Text cleaning and standardisation | `data/processed/posts_eda.parquet` | `data/processed/posts_clean.parquet` |
| `03_politeness_scoring.ipynb` | ConvoKit and DistilBERT scoring | `data/processed/posts_clean.parquet` | `data/processed/posts_scored.parquet` |
| `04_analysis.ipynb` | Comparative analysis and visualisation | `data/processed/posts_scored.parquet` | `data/processed/posts_analysis.parquet` |
| `05_causal_analysis.ipynb` | Back-door criterion and X-learner | `data/processed/posts_analysis.parquet` | `data/processed/posts_causal.parquet` |
| `06_llm_intervention_hf.ipynb` | LLM politeness rewriting feasibility | `data/processed/posts_scored.parquet` | `data/processed/posts_intervention.parquet` |

Run notebooks in order (01 through 06). All random seeds are fixed at 42.

---

## Dataset

The Stanford MOOC Forum Post dataset is sourced from the Stanford University Institute for Research in the Social Sciences. It contains 29,604 raw forum posts across 11 courses spanning three disciplines: Medicine, Humanities and Education. The dataset includes annotated scores for sentiment, confusion and urgency (1-7 scale) and binary flags for question, answer and opinion post types.

The raw dataset is not included in this repository. Please contact the Stanford University Institute for Research in the Social Sciences to request access.

---

## Accessing the Pipeline

Due to file size and GitHub rendering limitations, the repository additionally includes plain Python script exports of all six notebooks in the `scripts/` folder and HTML outputs of all executed notebooks in the `html/` folder, both of which render directly on GitHub without any additional tools.

As a backup, all files have also been uploaded to the following Google Drive folder:

**Google Drive:** https://drive.google.com/drive/folders/1lw6Y3D7ugJiIR2kUmAc5OkCPaTfGUQkc?usp=drive_link

All files were last modified on the day of submission. Notebooks can also be previewed via nbviewer at **https://nbviewer.org** by pasting the github ipynb url.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ikaranraman/politeness-causal-inference-mooc
cd politeness-causal-inference-mooc
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Place the raw dataset

Place `stanfordMOOCForumPostsSet.xlsx` in the `data/raw/` directory.

---

## Ethics

This study was approved by the Monash University Human Research Ethics Committee (MUHREC Project ID: 49024, approved 28 August 2025). The research involved secondary analysis of a publicly available, pre-anonymised dataset. No direct human participants were recruited and no personally identifiable information was used.

---

## Reproducibility

All random seeds are fixed at 42 throughout the pipeline. A `requirements.txt` is provided to recreate the Python environment. The repository contains a single clean commit reflecting the final state of the research pipeline.

---

## References

- Brown, P., & Levinson, S. C. (1987). Politeness: Some universals in language usage. Cambridge University Press.
- Danescu-Niculescu-Mizil, C. et al. (2013). A computational approach to politeness. ACL 2013.
- Kunzel, S. R. et al. (2019). Metalearners for estimating heterogeneous treatment effects. PNAS.
- Maathuis, M. H., & Colombo, D. (2015). A generalized back-door criterion. Annals of Statistics.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. arXiv.
