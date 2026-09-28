from typing import List, Tuple, Dict
import numpy as np
from warden_laya.model import LayaModelEngine


def evaluate_noul(
    engine: LayaModelEngine,
    query: str,
    candidates: List[Tuple[str, str]],
) -> List[Tuple[str, float, int]]:
    """
    Evaluates the 'noul' decision primitive: calibrated cross-attention
    probability P(Relevant | Query, Candidate Passage).

    Args:
        engine: LayaModelEngine instance.
        query: User search query string.
        candidates: List of (candidate_id, text) tuples.

    Returns:
        List of (candidate_id, calibrated_probability, 1-based rank),
        sorted in descending order of probability.
    """
    if not candidates:
        return []

    texts = [cand_text for _, cand_text in candidates]
    logits = engine.forward_batch(query=query, candidates=texts)

    # Compute binary softmax over [not_relevant, relevant]
    # P(rel) = exp(rel) / (exp(not_rel) + exp(rel))
    diff = logits[:, 1] - logits[:, 0]
    # Numerically stable sigmoid: 1 / (1 + exp(-diff))
    probs = 1.0 / (1.0 + np.exp(-np.clip(diff, -50.0, 50.0)))

    # Pair with candidate_ids and sort descending
    scored = []
    for (cand_id, _), prob in zip(candidates, probs):
        scored.append((cand_id, float(prob)))

    # Sort descending by probability
    scored.sort(key=lambda x: x[1], reverse=True)

    # Assign 1-based rank
    ranked = [(cand_id, prob, idx + 1) for idx, (cand_id, prob) in enumerate(scored)]
    return ranked


def evaluate_choice(
    engine: LayaModelEngine,
    query: str,
    choices: List[str],
) -> Tuple[str, Dict[str, float]]:
    """
    Evaluates the 'choice' decision primitive: categorical intent classification
    with normalized confidence distribution over discrete choices.

    Args:
        engine: LayaModelEngine instance.
        query: Incoming query string.
        choices: List of candidate choice labels (must contain >= 2 options).

    Returns:
        Tuple of (selected_choice_label, confidence_distribution_dict).
    """
    if len(choices) < 2:
        raise ValueError("choices must contain at least 2 distinct options")

    logits = engine.forward_choice(query=query, choices=choices)

    # Softmax normalization
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)

    # Build confidence distribution
    conf_dist = {choice: float(p) for choice, p in zip(choices, probs)}

    # Selected choice is the argmax
    best_idx = int(np.argmax(probs))
    selected_choice = choices[best_idx]

    return selected_choice, conf_dist


def evaluate_score(
    engine: LayaModelEngine,
    query: str,
    candidates: List[Tuple[str, str]],
) -> List[Tuple[str, float, int]]:
    """
    Evaluates the 'score' decision primitive: continuous relevance score.
    Returns (candidate_id, score, rank) sorted descending.
    """
    # In ModernBERT, score maps to the same calibrated cross-attention probability
    return evaluate_noul(engine, query, candidates)
