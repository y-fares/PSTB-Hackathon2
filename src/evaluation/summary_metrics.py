from typing import List, Dict

from rouge_score import rouge_scorer


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute simple ROUGE-1 and ROUGE-L scores between reference and hypothesis.

    This is a basic helper for measuring summary quality.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    # Extract F1 scores as a simple summary
    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_rouge_for_many(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute average ROUGE-1 and ROUGE-L F1 scores over many summaries.
    """
    if len(references) != len(hypotheses) or len(references) == 0:
        return {"rouge1_f1": 0.0, "rougeL_f1": 0.0}

    total_rouge1 = 0.0
    total_rougeL = 0.0
    n = len(references)

    for ref, hyp in zip(references, hypotheses):
        scores = compute_rouge_scores(ref, hyp)
        total_rouge1 += scores["rouge1_f1"]
        total_rougeL += scores["rougeL_f1"]

    return {
        "rouge1_f1": total_rouge1 / n,
        "rougeL_f1": total_rougeL / n,
    }