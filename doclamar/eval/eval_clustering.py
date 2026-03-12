import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def clustering_metrics(embeddings, predicted_labels=None, labels=None):
    results = {}
    if predicted_labels is not None:
        results['silhouette'] = float(silhouette_score(embeddings, predicted_labels))
        results['davies_bouldin'] = float(davies_bouldin_score(embeddings, predicted_labels))
        results['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, predicted_labels))
    if labels is not None and predicted_labels is not None:
        results['adjusted_rand'] = float(adjusted_rand_score(labels, predicted_labels))
        results['nmi'] = float(normalized_mutual_info_score(labels, predicted_labels))
    return results

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', default='doclamar/eval/data/sample_embeddings.npy')
    parser.add_argument('--pred', default='doclamar/eval/data/sample_pred_labels.json')
    parser.add_argument('--labels', default=None)
    args = parser.parse_args()
    emb = np.load(args.emb)
    pred = json.load(open(args.pred))
    pred_labels = pred.get('labels')
    labels = None
    if args.labels:
        labels = json.load(open(args.labels)).get('labels')
    res = clustering_metrics(emb, predicted_labels=pred_labels, labels=labels)
    print('Clustering metrics:')
    print(json.dumps(res, indent=2))
