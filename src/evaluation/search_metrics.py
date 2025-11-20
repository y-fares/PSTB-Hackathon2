from typing import List, Set


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute precision@k.

    retrieved: list of retrieved indices (e.g., chunk indices).
    relevant: set of relevant indices.
    k: how many items we consider (usually len(retrieved) or smaller).

    Precision@k = (# of relevant items in top k) / k
    """
    if k == 0:
        return 0.0

    top_k = retrieved[:k]
    num_relevant_in_top_k = sum(1 for idx in top_k if idx in relevant)
    precision = num_relevant_in_top_k / float(k)
    return precision


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Compute recall@k.

    Recall@k = (# of relevant items in top k) / (# of all relevant items)
    """
    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    num_relevant_in_top_k = sum(1 for idx in top_k if idx in relevant)
    recall = num_relevant_in_top_k / float(len(relevant))
    return recall