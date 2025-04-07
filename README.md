# A systematic evaluation of hybrid retrieval strategies

This project compares individual (BM25, SPLADE, DPR, and ColBERT) and hybrid (Score Fusion, Sequential Retrieval, and Union Retrieval) information retrieval methods, evaluated on the CISI dataset.

## Research Questions

1. How do individual retrieval models (BM25, SPLADE, DPR, ColBERT) perform when used independently for document retrieval on the CISI dataset when evaluated using precision and MAP?
2. What hybrid method between score fusion, sequential retrieval with two-stage reranking, and union retrieval performs the best on the CISI dataset?
3. How effective are hybrid retrieval strategies on the CISI dataset compared to individual models?

## Motivation

Traditional lexical models (like BM25) and neural models (like DPR and ColBERT) have complementary strengths. Hybrid IR aims to combine these to improve overall performance.

## Dataset

We use the [CISI dataset](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval), which contains:

- 1460 documents in information science
- 112 keyword-based queries
- Binary relevance labels

## Retrieval Models

### Individual
- **BM25** (from `rank_bm25`)
- **SPLADE** 
- **DPR** 
- **ColBERT**

### Hybrid
- **Score Fusion**: Weighted sum of scores from two retrievers
- **Sequential Retrieval**: First model retrieves, second re-ranks
- **Union Retrieval**: Combined candidate sets with re-ranking

## Evaluation

- Metrics: Precision, Mean Average Precision (MAP)
- Significance Testing: Wilcoxon signed-rank test

## Installation

```bash
git clone https://github.com/adina-dobrinoiu/Information-Retrieval.git
cd Information-Retrieval
pip install -r requirements.txt
