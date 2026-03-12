import json
from typing import Dict, List

def tokenize(s: str):
    return s.split()

def chunk_f1(gt_text: str, pred_text: str) -> float:
    g = set(tokenize(gt_text))
    p = set(tokenize(pred_text))
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    prec = tp / len(p) if len(p) else 0.0
    rec = tp / len(g) if len(g) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def parsing_accuracy(gt_map, pred_map, match_threshold=0.5):
    total_gt = 0
    correct = 0
    for doc, gt_chunks in gt_map.items():
        preds = pred_map.get(doc, [])
        for g in gt_chunks:
            total_gt += 1
            best = 0.0
            for p in preds:
                f1 = chunk_f1(g.get('text',''), p.get('text',''))
                if f1 > best:
                    best = f1
            if best >= match_threshold:
                correct += 1
    return correct / max(1, total_gt)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='doclamar/eval/data/sample_gt_parsing.json')
    parser.add_argument('--pred', default='doclamar/eval/data/sample_pred_parsing.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    gt = load_json(args.gt)
    pred = load_json(args.pred)
    acc = parsing_accuracy(gt, pred, args.threshold)
    print(f'Parsing accuracy (threshold={args.threshold}):', acc)
