"""Produce a publication-ready CSV/Markdown table of metrics.

This script runs the available metric functions, computes bootstrap CIs,
and writes `eval_results.csv` and `eval_results.md` in `doclamar/eval/`.

It also computes simple baselines (random or heuristic) so you have comparison numbers.
"""
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'eval' / 'data'
OUT = ROOT / 'eval'

from doclamar.eval.eval_mrr import mean_reciprocal_rank
from doclamar.eval.eval_parsing import parsing_accuracy
from doclamar.eval.eval_clustering import clustering_metrics
from doclamar.eval.eval_tasks import task_success_rates
from doclamar.eval.eval_rouge import rouge_for_corpus
from doclamar.eval.bootstrap import bootstrap_metric_per_item

def per_query_rr(gt_map, res_map):
    # return list of reciprocal ranks per query
    rrs = []
    for qid, gt in gt_map.items():
        retrieved = res_map.get(qid, [])
        rr = 0.0
        for i, doc in enumerate(retrieved, start=1):
            if doc in set(gt):
                rr = 1.0 / i
                break
        rrs.append(rr)
    return rrs

def random_baseline_mrr(gt_map, res_map, n_trials=50, seed=0):
    rng = random.Random(seed)
    docs = set()
    for lst in res_map.values():
        docs.update(lst)
    docs = list(docs) or ['d1','d2','d3']
    trial_means = []
    for _ in range(n_trials):
        rand_res = {q: rng.sample(docs, min(len(docs), len(res_map.get(q, docs)))) for q in gt_map}
        trial_means.append(float(np.mean(per_query_rr(gt_map, rand_res))))
    return float(np.mean(trial_means))

def parsing_baseline(gt_map, pred_map):
    # baseline: naive sentence splitter (split on '.')
    baseline_pred = {}
    for doc, gchunks in gt_map.items():
        # attempt to load the original document text from knowledge if available
        # fallback: join gt chunk texts
        text = ' '.join([c.get('text','') for c in gchunks])
        sents = [s.strip() for s in text.split('.') if s.strip()]
        baseline_pred[doc] = [{'text': s} for s in sents]
    return parsing_accuracy(gt_map, baseline_pred)

def clustering_random_baseline(emb, n_clusters=5, seed=0):
    rng = np.random.RandomState(seed)
    n = emb.shape[0]
    labels = rng.randint(0, n_clusters, size=n).tolist()
    return clustering_metrics(emb, predicted_labels=labels)

def rouge_random_baseline(refs, hyps, n_trials=50, seed=0):
    rng = random.Random(seed)
    scores = []
    for _ in range(n_trials):
        shuffled = hyps[:] 
        rng.shuffle(shuffled)
        scores.append(rouge_for_corpus(refs, shuffled))
    # average across trials
    avg = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    return avg

def make_table():
    # Load available sample data (these are the fallback/samples we added earlier)
    gt_mrr = json.load(open(DATA / 'sample_gt_mrr.json'))
    res_mrr = json.load(open(DATA / 'sample_res_mrr.json'))

    # MRR and bootstrap
    per_item = per_query_rr(gt_mrr, res_mrr)
    mrr_mean, mrr_low, mrr_high = bootstrap_metric_per_item(per_item, n_iter=500, seed=1)

    # Random baseline MRR
    mrr_rand = random_baseline_mrr(gt_mrr, res_mrr)

    # Parsing
    gt_par = json.load(open(DATA / 'sample_gt_parsing.json'))
    pred_par = json.load(open(DATA / 'sample_pred_parsing.json'))
    # compute per-chunk match F1 as proxy: for each gt chunk, best-match f1
    per_chunk_scores = []
    from doclamar.eval.eval_parsing import chunk_f1
    for doc, gchunks in gt_par.items():
        preds = pred_par.get(doc, [])
        for g in gchunks:
            best = 0.0
            for p in preds:
                best = max(best, chunk_f1(g.get('text',''), p.get('text','')))
            per_chunk_scores.append(best)
    par_mean, par_low, par_high = bootstrap_metric_per_item(per_chunk_scores, n_iter=500, seed=2)
    par_baseline = parsing_baseline(gt_par, pred_par)

    # Clustering
    emb_path = DATA / 'sample_embeddings.npy'
    if not emb_path.exists():
        # trigger creation via runner
        import numpy as _np
        emb = _np.random.RandomState(0).randn(100, 64)
        _np.save(emb_path, emb)
    emb = np.load(emb_path)
    # predicted labels file
    pred_labels = json.load(open(DATA / 'sample_pred_labels.json'))
    pred = pred_labels.get('labels')
    clust_res = clustering_metrics(emb, predicted_labels=pred)
    clust_rand = clustering_random_baseline(emb, n_clusters=5)

    # Tasks
    runs = json.load(open(DATA / 'sample_runs.json'))
    overall_success, per_task = task_success_rates(runs)

    # ROUGE
    refs = json.load(open(DATA / 'sample_refs.json'))
    hyps = json.load(open(DATA / 'sample_hyps.json'))
    rouge_res = rouge_for_corpus(refs, hyps)
    rouge_rand = rouge_random_baseline(refs, hyps)

    # Assemble table rows
    rows = []
    rows.append({'Metric Category':'Retrieval','Metric':'MRR','Value':mrr_mean,'CI_low':mrr_low,'CI_high':mrr_high,'Baseline':'Random','Baseline_Value':mrr_rand})
    rows.append({'Metric Category':'Parsing','Metric':'Chunk-F1 (avg)','Value':par_mean,'CI_low':par_low,'CI_high':par_high,'Baseline':'Sentence-split','Baseline_Value':par_baseline})
    rows.append({'Metric Category':'Clustering','Metric':'Silhouette','Value':clust_res.get('silhouette'),'CI_low':'-','CI_high':'-','Baseline':'Random labels','Baseline_Value':clust_rand.get('silhouette')})
    rows.append({'Metric Category':'Clustering','Metric':'Adjusted Rand','Value':clust_res.get('adjusted_rand'),'CI_low':'-','CI_high':'-','Baseline':'Random labels','Baseline_Value':clust_rand.get('adjusted_rand')})
    rows.append({'Metric Category':'Pipeline','Metric':'Overall Run Success Rate','Value':overall_success,'CI_low':'-','CI_high':'-','Baseline':'N/A','Baseline_Value':None})
    rows.append({'Metric Category':'Summarization','Metric':'ROUGE-1','Value':rouge_res['rouge1'],'CI_low':'-','CI_high':'-','Baseline':'Random hyps','Baseline_Value':rouge_rand['rouge1']})
    rows.append({'Metric Category':'Summarization','Metric':'ROUGE-2','Value':rouge_res['rouge2'],'CI_low':'-','CI_high':'-','Baseline':'Random hyps','Baseline_Value':rouge_rand['rouge2']})
    rows.append({'Metric Category':'Summarization','Metric':'ROUGE-L','Value':rouge_res['rougeL'],'CI_low':'-','CI_high':'-','Baseline':'Random hyps','Baseline_Value':rouge_rand['rougeL']})

    df = pd.DataFrame(rows)
    csv_path = OUT / 'eval_results.csv'
    md_path = OUT / 'eval_results.md'
    df.to_csv(csv_path, index=False)
    # simple markdown
    try:
        md = df.to_markdown(index=False)
    except Exception:
        # fallback: simple markdown table
        cols = list(df.columns)
        header = '| ' + ' | '.join(cols) + ' |\n'
        sep = '| ' + ' | '.join(['---']*len(cols)) + ' |\n'
        rows_txt = ''
        for _, r in df.iterrows():
            rows_txt += '| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n'
        md = header + sep + rows_txt
    md_path.write_text(md)
    print('Wrote', csv_path, md_path)

if __name__ == '__main__':
    make_table()
