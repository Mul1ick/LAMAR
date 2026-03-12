"""Unified runner that demonstrates all evaluation scripts on sample data.

Replace sample JSON files in `doclamar/eval/data/` with your real outputs and ground truth.
"""
import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'eval' / 'data'

def run_mrr():
    from doclamar.eval.eval_mrr import mean_reciprocal_rank
    gt = json.load(open(DATA / 'sample_gt_mrr.json'))
    res = json.load(open(DATA / 'sample_res_mrr.json'))
    print('MRR:', mean_reciprocal_rank(gt, res))

def run_parsing():
    from doclamar.eval.eval_parsing import parsing_accuracy
    gt = json.load(open(DATA / 'sample_gt_parsing.json'))
    pred = json.load(open(DATA / 'sample_pred_parsing.json'))
    print('Parsing accuracy (threshold=0.5):', parsing_accuracy(gt, pred, 0.5))

def run_clustering():
    import numpy as _np
    from doclamar.eval.eval_clustering import clustering_metrics
    emb_path = DATA / 'sample_embeddings.npy'
    if not emb_path.exists():
        # create synthetic embeddings and pred labels for demo
        emb = _np.random.RandomState(0).randn(100, 64)
        _np.save(emb_path, emb)
        pred = {'labels': [int(i%5) for i in range(100)]}
        json.dump(pred, open(DATA / 'sample_pred_labels.json','w'))
    emb = _np.load(emb_path)
    pred = json.load(open(DATA / 'sample_pred_labels.json'))
    res = clustering_metrics(emb, predicted_labels=pred['labels'])
    print('Clustering metrics:')
    print(json.dumps(res, indent=2))

def run_tasks():
    from doclamar.eval.eval_tasks import task_success_rates
    runs = json.load(open(DATA / 'sample_runs.json'))
    overall, per_task = task_success_rates(runs)
    print('Overall run success rate:', overall)
    print('Per-task success rates:')
    print(json.dumps(per_task, indent=2))

def run_rouge():
    from doclamar.eval.eval_rouge import rouge_for_corpus
    refs = json.load(open(DATA / 'sample_refs.json'))
    hyps = json.load(open(DATA / 'sample_hyps.json'))
    print('ROUGE scores:')
    print(rouge_for_corpus(refs, hyps))

def main():
    print('\n--- Running MRR ---')
    run_mrr()
    print('\n--- Running Parsing Accuracy ---')
    run_parsing()
    print('\n--- Running Clustering Metrics ---')
    run_clustering()
    print('\n--- Running Task Success Rates ---')
    run_tasks()
    print('\n--- Running ROUGE ---')
    run_rouge()

if __name__ == '__main__':
    main()
