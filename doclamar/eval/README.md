Evaluation toolkit for Doclamar

This folder contains evaluation scripts for the metrics requested:
- Mean Reciprocal Rank (MRR)
- Parsing accuracy (span/token F1 and exact match)
- Clustering metrics (silhouette, davies_bouldin, ARI, NMI)
- Task success rate (pipeline runs)
- ROUGE scores (rouge1, rouge2, rougeL)

Quick start
1. Install dependencies:

```
pip install -r doclamar/eval/requirements_eval.txt
```

2. Run the unified runner which demonstrates all metrics on sample data:

```
python doclamar/eval/eval_runner.py
```

The runner will load sample JSON files in `doclamar/eval/data/`. Replace those with your real outputs and ground-truth files and re-run.
