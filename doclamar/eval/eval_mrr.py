import json
from typing import Dict, List, Set

def reciprocal_rank(gt: Set[str], retrieved: List[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in gt:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(gt_map: Dict[str, Set[str]], res_map: Dict[str, List[str]]) -> float:
    total = 0.0
    count = 0
    for qid, gt in gt_map.items():
        retrieved = res_map.get(qid, [])
        total += reciprocal_rank(set(gt), retrieved)
        count += 1
    return total / max(1, count)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='doclamar/eval/data/sample_gt_mrr.json')
    parser.add_argument('--res', default='doclamar/eval/data/sample_res_mrr.json')
    args = parser.parse_args()
    gt = load_json(args.gt)
    res = load_json(args.res)
    print('MRR:', mean_reciprocal_rank(gt, res))
