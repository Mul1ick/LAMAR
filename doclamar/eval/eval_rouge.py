from statistics import mean

try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    def rouge_for_corpus(refs, hyps):
        scores = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
        for r, h in zip(refs, hyps):
            s = scorer.score(r, h)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        return {k: mean(v) if v else 0.0 for k,v in scores.items()}
except Exception:
    # Lightweight fallback: compute ROUGE-1/2 via ngram overlap, ROUGE-L via LCS ratio
    import re

    def tokenize(s):
        return re.findall(r"\w+", s.lower())

    def ngram_counts(tokens, n):
        from collections import Counter
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def rouge_n(ref, hyp, n):
        r_tokens = tokenize(ref)
        h_tokens = tokenize(hyp)
        rc = ngram_counts(r_tokens, n)
        hc = ngram_counts(h_tokens, n)
        overlap = sum(min(rc[g], hc.get(g,0)) for g in rc)
        prec = overlap / sum(hc.values()) if sum(hc.values())>0 else 0.0
        rec = overlap / sum(rc.values()) if sum(rc.values())>0 else 0.0
        if prec+rec == 0:
            return 0.0
        return 2*prec*rec/(prec+rec)

    def lcs_length(a, b):
        # dynamic programming LCS length
        la, lb = len(a), len(b)
        dp = [[0]*(lb+1) for _ in range(la+1)]
        for i in range(la-1, -1, -1):
            for j in range(lb-1, -1, -1):
                if a[i]==b[j]:
                    dp[i][j] = 1 + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1])
        return dp[0][0]

    def rouge_l(ref, hyp):
        r_tokens = tokenize(ref)
        h_tokens = tokenize(hyp)
        if not r_tokens or not h_tokens:
            return 0.0
        lcs = lcs_length(r_tokens, h_tokens)
        prec = lcs / len(h_tokens)
        rec = lcs / len(r_tokens)
        if prec+rec == 0:
            return 0.0
        return 2*prec*rec/(prec+rec)

    def rouge_for_corpus(refs, hyps):
        scores = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
        for r,h in zip(refs, hyps):
            scores['rouge1'].append(rouge_n(r,h,1))
            scores['rouge2'].append(rouge_n(r,h,2))
            scores['rougeL'].append(rouge_l(r,h))
        return {k: mean(v) if v else 0.0 for k,v in scores.items()}

if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', default='doclamar/eval/data/sample_refs.json')
    parser.add_argument('--hyps', default='doclamar/eval/data/sample_hyps.json')
    args = parser.parse_args()
    refs = json.load(open(args.refs))
    hyps = json.load(open(args.hyps))
    res = rouge_for_corpus(refs, hyps)
    print('ROUGE scores:')
    print(res)
