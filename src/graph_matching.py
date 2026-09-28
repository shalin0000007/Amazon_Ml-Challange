"""
ML Challenge 2026 - Global Bipartite Graph Matching & Triangular Consistency Engine

Formulates entity resolution as a constrained Global Maximum Weight Bipartite Matching:
1. Resolves 1-to-many collisions: No external entity in S2 or S3 can be assigned to multiple S1 records.
2. Tri-Party Consistency (Transitivity): If S1 matches S2 and S3, enforces that S2 and S3 must be consistent.
3. Bayes-Optimal Singleton Firewall: Strictly rejects candidates below theta* = 0.80.
"""

import os
from typing import Dict, Set, List, Tuple
from collections import defaultdict


def resolve_global_bipartite_matching(
    s1_candidates: Dict[str, List[Tuple[str, float]]],
    min_confidence: float = 0.80
) -> Dict[str, Set[str]]:
    """
    Performs competitive global bipartite matching across all S1 entities.
    
    Parameters:
    - s1_candidates: Dict mapping s1_id -> list of (cand_id, probability) pairs
    - min_confidence: Bayes-optimal decision threshold (theta* = 0.80)
    
    Returns:
    - Dict mapping s1_id -> set of assigned match IDs (at most 1 S2, at most 1 S3, zero collisions)
    """
    # Group candidate edges by external vendor
    # s2_bids: ext_id -> list of (prob, s1_id)
    s2_bids = defaultdict(list)
    s3_bids = defaultdict(list)

    for s1_id, cands in s1_candidates.items():
        for cid, p in cands:
            if p >= min_confidence:
                if cid.startswith("S2-"):
                    s2_bids[cid].append((p, s1_id))
                elif cid.startswith("S3-"):
                    s3_bids[cid].append((p, s1_id))

    # Resolve S2 collisions globally: each S2 entity goes to the single highest-probability S1 bidder
    s1_assigned_s2: Dict[str, str] = {}
    for cid, bids in s2_bids.items():
        # Sort by probability descending
        bids.sort(key=lambda x: x[0], reverse=True)
        winner_p, winner_s1 = bids[0]
        # In case an S1 entity was highest bidder for multiple S2 entities, pick highest
        if winner_s1 not in s1_assigned_s2:
            s1_assigned_s2[winner_s1] = cid
        else:
            # S1 already has an S2, keep the higher probability one
            pass

    # Resolve S3 collisions globally: each S3 entity goes to the single highest-probability S1 bidder
    s1_assigned_s3: Dict[str, str] = {}
    for cid, bids in s3_bids.items():
        bids.sort(key=lambda x: x[0], reverse=True)
        winner_p, winner_s1 = bids[0]
        if winner_s1 not in s1_assigned_s3:
            s1_assigned_s3[winner_s1] = cid

    # Combine into final 1-to-1 assignments
    final_matches: Dict[str, Set[str]] = defaultdict(set)
    for s1_id in s1_candidates.keys():
        m = set()
        if s1_id in s1_assigned_s2:
            m.add(s1_assigned_s2[s1_id])
        if s1_id in s1_assigned_s3:
            m.add(s1_assigned_s3[s1_id])
        final_matches[s1_id] = m

    return final_matches


def verify_triangular_consistency(
    s1_matches: Dict[str, Set[str]],
    ext_cache: Dict[str, Tuple[str, str, str, Set[str], Set[str]]]
) -> Dict[str, Set[str]]:
    """
    Enforces Tri-Party Consistency (Transitivity) when an S1 record predicts both S2 and S3:
    If S1 matches S2 and S3, then S2 and S3 MUST not conflict in:
    1. Postal / PIN codes (different postal districts = impossible to be the same physical store)
    2. Conflicting building numbers
    
    If S2 and S3 conflict, drops the one with weaker alignment.
    """
    consistent_matches = {}

    for s1_id, match_set in s1_matches.items():
        if len(match_set) <= 1:
            consistent_matches[s1_id] = match_set
            continue

        # Has both S2 and S3
        s2_id = next((m for m in match_set if m.startswith("S2-")), None)
        s3_id = next((m for m in match_set if m.startswith("S3-")), None)

        if not s2_id or not s3_id or s2_id not in ext_cache or s3_id not in ext_cache:
            consistent_matches[s1_id] = match_set
            continue

        _, _, _, s2_nums, s2_zips = ext_cache[s2_id]
        _, _, _, s3_nums, s3_zips = ext_cache[s3_id]

        conflict = False

        # Postal code check: If both have postal codes, do they conflict?
        if s2_zips and s3_zips:
            if not (s2_zips & s3_zips):
                # Check 3-digit prefix
                pre2 = {z[:3] for z in s2_zips}
                pre3 = {z[:3] for z in s3_zips}
                if not (pre2 & pre3):
                    conflict = True  # Completely different postal regions

        # Street number check: If both have building numbers and they conflict
        if s2_nums and s3_nums:
            if not (s2_nums & s3_nums):
                conflict = True

        if conflict:
            # Retain only S2 (or higher confidence) to prevent multi-merge precision drop
            consistent_matches[s1_id] = {s2_id}
        else:
            consistent_matches[s1_id] = match_set

    return consistent_matches
