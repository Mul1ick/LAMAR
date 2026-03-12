import json

def task_success_rates(runs):
    per_task = {}
    total_runs = len(runs)
    run_successes = 0
    for run in runs:
        tasks = run.get('tasks', [])
        run_ok = all(t.get('status') == 'success' for t in tasks)
        if run_ok:
            run_successes += 1
        for t in tasks:
            name = t.get('name')
            stat = per_task.setdefault(name, {'success':0, 'total':0})
            stat['total'] += 1
            if t.get('status') == 'success':
                stat['success'] += 1
    per_task_rates = {name: v['success']/v['total'] for name, v in per_task.items()}
    overall_run_success_rate = run_successes / max(1, total_runs)
    return overall_run_success_rate, per_task_rates

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='doclamar/eval/data/sample_runs.json')
    args = parser.parse_args()
    runs = json.load(open(args.runs))
    overall, per_task = task_success_rates(runs)
    print('Overall run success rate:', overall)
    print('Per-task success rates:')
    import json as _j
    print(_j.dumps(per_task, indent=2))
