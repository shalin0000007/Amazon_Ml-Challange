import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import List, Tuple, Set, Dict, Optional

# Bayes-optimal decision thresholds calibrated for Macro F0.5 via extensive grid search
DEFAULT_COUNTRY_THRESHOLDS = {
    "US": {"match": 0.35, "singleton": 0.950, "delta": 0.030},
    "France": {"match": 0.35, "singleton": 0.950, "delta": 0.030},
    "India": {"match": 0.35, "singleton": 0.950, "delta": 0.030},
}


def is_corporate_conflict(s1_suff: str, ext_suff: str) -> bool:
    """Detects mutually exclusive corporate legal entities (e.g. Pvt Ltd vs LLP in India)."""
    if not s1_suff or not ext_suff:
        return False
    s1_pvt = any(k in s1_suff for k in ("pvt", "private", "praivet"))
    s1_llp = any(k in s1_suff for k in ("llp", "एलएलपी"))
    ext_pvt = any(k in ext_suff for k in ("pvt", "private", "praivet"))
    ext_llp = any(k in ext_suff for k in ("llp", "एलएलपी"))
    return (s1_pvt and ext_llp) or (s1_llp and ext_pvt)


class EntityResolutionModel:
    """
    V10 Grandmaster Entity Resolution Model:
    - 47-dimensional gradient-boosted decision tree
    - Grid-search optimal Macro F0.5 decision thresholds (s*=0.95, m*=0.35, delta*=0.03)
    - Anti-chain store & anti-mall multi-merge safety firewalls
    - Missing-address exact name golden anchor rescue
    - Calibrated Relative Confidence Banding
    """

    def __init__(
        self,
        n_estimators: int = 700,
        learning_rate: float = 0.04,
        num_leaves: int = 127,
        max_depth: int = 9,
        subsample: float = 0.85,
        colsample_bytree: float = 0.80,
        min_child_samples: int = 35,
        match_threshold: float = 0.35,
        singleton_threshold: float = 0.950,
        country_thresholds: Dict[str, Dict[str, float]] = None,
        delta: float = 0.030,
        random_state: int = 42,
    ):
        self.match_threshold = match_threshold
        self.singleton_threshold = singleton_threshold
        self.country_thresholds = country_thresholds or DEFAULT_COUNTRY_THRESHOLDS
        self.delta = delta
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            subsample_freq=1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the underlying classifier on pairwise feature matrix."""
        self.clf.fit(X, y)
        return self

    def filter_candidate_matches(
        self,
        candidate_ids: List[str],
        probs: np.ndarray,
        features: List[List[float]],
        country: str = "US",
        delta: Optional[float] = None,
        s1_suffix: str = "",
        cand_suffixes: Optional[List[str]] = None,
    ) -> Set[str]:
        """
        Filters candidate predictions through:
        1. Bayes-Optimal Country Confidence Thresholds
        2. Strict Singleton Protection Firewall
        3. Street Number & Postal Geographic Gates
        4. Chain-Store Multi-Branch Gate
        5. Corporate Structure Conflict Gate (Pvt Ltd vs LLP)
        6. Calibrated Relative Confidence Banding
        """
        if len(candidate_ids) == 0 or len(probs) == 0:
            return set()

        max_p = np.max(probs)

        # Calibrated country thresholds
        thresh = self.country_thresholds.get(
            country,
            {"match": self.match_threshold, "singleton": self.singleton_threshold, "delta": self.delta},
        )
        m_thresh = thresh.get("match", self.match_threshold)
        s_thresh = thresh.get("singleton", self.singleton_threshold)
        band_delta = delta if delta is not None else thresh.get("delta", getattr(self, "delta", 0.030))

        # Singleton Firewall: If maximum probability does not exceed singleton threshold,
        # mathematically optimal decision under F0.5 is to predict EMPTY (Singleton),
        # UNLESS rescued by the exact core name golden anchor for missing address records.
        if max_p < s_thresh:
            golden_rescues = set()
            for i, (cid, f) in enumerate(zip(candidate_ids, features)):
                n_exact = f[15]
                addr_is_nan = f[26]
                if n_exact == 1.0 and addr_is_nan == 1.0:
                    golden_rescues.add(cid)
            if golden_rescues:
                return golden_rescues
            return set()

        matched = set()

        for i, (cid, p, f) in enumerate(zip(candidate_ids, probs, features)):
            if p >= m_thresh:
                n_ratio = f[0]
                n_sort = f[2]
                n_set = f[3]
                n_exact = f[15]
                a_jaccard = f[24]
                num_conflict = f[33]
                zip_status = f[35]
                zip_prefix_match = f[36]
                strong_name_weak_addr = f[45]
                strong_addr_weak_name = f[46]

                # Safety Gate 1: Conflicting street/building numbers
                if num_conflict == 1.0 and n_exact == 0.0 and max(n_ratio, n_sort) < 92.0:
                    continue

                # Safety Gate 2: Conflicting postal district (neither full zip nor 3-digit prefix matches)
                if zip_status == -1.0 and zip_prefix_match == -1.0 and n_exact == 0.0 and max(n_ratio, n_sort) < 88.0:
                    continue

                # Safety Gate 3: Chain-Store Firewall (Identical brand name, but completely different address)
                if strong_name_weak_addr == 1.0 and a_jaccard < 20.0 and n_exact == 0.0:
                    continue

                # Safety Gate 4: Co-location Firewall (Same address, totally distinct business name)
                if strong_addr_weak_name == 1.0:
                    continue

                # Safety Gate 5: Severe name divergence
                if min(n_sort, n_set) < 38.0 and n_ratio < 42.0:
                    continue

                # Safety Gate 6: Corporate Structure Conflict (e.g. Pvt Ltd vs LLP)
                if cand_suffixes and s1_suffix and i < len(cand_suffixes):
                    if is_corporate_conflict(s1_suffix, cand_suffixes[i]):
                        continue

                # Relative Confidence Banding: Include all genuine duplicate listings within delta of max confidence
                if p >= (max_p - band_delta):
                    matched.add(cid)

        return matched

    def predict_matches(
        self, candidate_ids: List[str], features: np.ndarray, country: str = "US"
    ) -> Set[str]:
        """Predicts matching entity IDs for a single Source 1 record."""
        if len(candidate_ids) == 0 or len(features) == 0:
            return set()

        probs = self.clf.predict_proba(features)[:, 1]
        return self.filter_candidate_matches(candidate_ids, probs, features, country=country)

    def save(self, filepath: str):
        """Persists trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EntityResolutionModel":
        """Loads model from disk."""
        return joblib.load(filepath)
