from tests.evaluator import run_eval_suite, print_eval_report

TEST_CASES = [
    {
        "query": "What is machine learning?",
        "reference_answer": "Machine learning is a branch of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        "relevant_files": ["Machine learning.txt"],
    },
    {
        "query": "Explain the autoencoder architecture used in lip reading",
        "reference_answer": "The autoencoder extracts bottleneck features from auditory spectrograms which are used as targets for the main lip reading network.",
        "relevant_files": ["Lip_reading.txt"],
    },
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", "-p", required=True)
    parser.add_argument("--top_k", "-k", type=int, default=5)
    args = parser.parse_args()

    results = run_eval_suite(TEST_CASES, args.root_path, args.top_k)
    print_eval_report(results)
