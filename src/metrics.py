"""Exact evaluation metric implementation for Amazon ML Challenge 2026.

Macro-averaged F_0.5 Score:
- F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Singletons: If true matches == empty and predicted matches == empty -> 1.0;
              If true matches == empty and predicted matches != empty -> 0.0.
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity."""
    n_gt = len(ground_truth_ids)
    n_pred = len(predicted_ids)

    # Singleton case (0 true matches)
    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0

    # No prediction for a non-singleton entity
    if n_pred == 0:
        return 0.0

    true_positives = len(predicted_ids & ground_truth_ids)
    if true_positives == 0:
        return 0.0

    precision = true_positives / n_pred
    recall = true_positives / n_gt

    numerator = 1.25 * precision * recall
    denominator = 0.25 * precision + recall

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Iterable[str] = None
) -> Dict[str, float]:
    """Computes overall macro F_0.5 score across all Source 1 entities."""
    if all_s1_ids is None:
        all_s1_ids = ground_truth.keys()

    total_f05 = 0.0
    singleton_f05 = 0.0
    non_singleton_f05 = 0.0
    n_total = 0
    n_singletons = 0
    n_non_singletons = 0

    for s1_id in all_s1_ids:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, gt_set)

        total_f05 += score
        n_total += 1

        if len(gt_set) == 0:
            singleton_f05 += score
            n_singletons += 1
        else:
            non_singleton_f05 += score
            n_non_singletons += 1

    macro_score = total_f05 / max(1, n_total)
    singleton_score = singleton_f05 / max(1, n_singletons) if n_singletons else 0.0
    non_singleton_score = non_singleton_f05 / max(1, n_non_singletons) if n_non_singletons else 0.0

    return {
        "macro_f05": macro_score,
        "singleton_f05": singleton_score,
        "non_singleton_f05": non_singleton_score,
        "n_entities": n_total,
        "n_singletons": n_singletons,
        "n_non_singletons": n_non_singletons,
    }
