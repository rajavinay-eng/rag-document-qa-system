import time
import statistics
import numpy as np
import chromadb

client = chromadb.PersistentClient(path="./project2_db")

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def evaluate_faithfulness(answer, contexts):

    if not answer or not contexts:
        return 0.0

    answer_words = set(answer.lower().split())

    context_words = set(
        " ".join(contexts).lower().split()
    )

    overlap = answer_words.intersection(
        context_words
    )

    return round(
        len(overlap) / max(len(answer_words), 1),
        3
    )

def evaluate_answer_relevancy(question, answer):

    if not question or not answer:
        return 0.0

    q_words = set(question.lower().split())

    a_words = set(answer.lower().split())

    overlap = q_words.intersection(a_words)

    return round(
        len(overlap) / max(len(q_words), 1),
        3
    )

def evaluate_context_precision(question, contexts):

    if not contexts:
        return 0.0

    q_words = set(question.lower().split())

    scores = []

    for context in contexts:

        c_words = set(context.lower().split())

        overlap = q_words.intersection(c_words)

        score = len(overlap) / max(len(q_words), 1)

        scores.append(score)

    return round(float(np.mean(scores)), 3)

def evaluate_pipeline(question, answer, contexts):

    return {
        "faithfulness": evaluate_faithfulness(
            answer,
            contexts
        ),

        "answer_relevancy": evaluate_answer_relevancy(
            question,
            answer
        ),

        "context_precision": evaluate_context_precision(
            question,
            contexts
        )
    }

def run_full_evaluation(test_cases):

    results = []

    for case in test_cases:

        scores = evaluate_pipeline(
            case["question"],
            case["answer"],
            case["contexts"]
        )

        results.append(scores)

    return {

        "faithfulness": round(
            np.mean([
                r["faithfulness"]
                for r in results
            ]),
            3
        ),

        "answer_relevancy": round(
            np.mean([
                r["answer_relevancy"]
                for r in results
            ]),
            3
        ),

        "context_precision": round(
            np.mean([
                r["context_precision"]
                for r in results
            ]),
            3
        ),

        "n_samples": len(results)
    }

def measure_latency(question, n_runs=3):

    timings = []

    for _ in range(n_runs):

        start = time.time()

        collection.query(
            query_texts=[question],
            n_results=1
        )

        elapsed = (
            time.time() - start
        ) * 1000

        timings.append(elapsed)

    return {

        "p50_ms": round(
            statistics.median(timings),
            1
        ),

        "p95_ms": round(
            max(timings),
            1
        )
    }

if __name__ == "__main__":

    print("evaluation.py working")