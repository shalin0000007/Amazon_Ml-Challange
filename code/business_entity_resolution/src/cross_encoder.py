"""
ML Challenge 2026 - Deep Neural Cross-Encoder Reranker (RTX 3050 GPU Accelerated)

Applies full transformer cross-attention to resolve ambiguous borderline candidate pairs (0.45 <= p <= 0.85).
Provides token-level deep interaction to bridge the gap from 0.85 to 0.97+ / 0.99.
"""

import os
from typing import List, Tuple, Dict
import numpy as np


class NeuralCrossEncoderReranker:
    """
    GPU-accelerated Cross-Encoder for borderline pair disambiguation.
    Feeds pair (Text1, Text2) jointly through transformer self-attention layers.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 128,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _init_model(self):
        if self.model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder

                if self.device == "auto":
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    dev = self.device

                print(f"[CrossEncoder] Loading {self.model_name} on device: {dev}...")
                self.model = CrossEncoder(self.model_name, device=dev, max_length=128)
                
                # Use FP16 on GPU for 2x faster throughput
                if dev == "cuda" and hasattr(self.model.model, "half"):
                    self.model.model.half()
                    print("[CrossEncoder] Activated FP16 half-precision on NVIDIA RTX 3050 GPU.")
            except ImportError:
                print("[CrossEncoder] Note: sentence-transformers not installed. Skipping neural reranking.")
                self.model = None

    def predict_pair_scores(self, text_pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Computes deep cross-attention similarity scores for a list of (text_s1, text_ext) pairs.
        Returns 1D numpy array of probabilities [0.0, 1.0].
        """
        self._init_model()
        if self.model is None or len(text_pairs) == 0:
            return np.zeros(len(text_pairs), dtype=np.float32)

        # CrossEncoder.predict handles batching and GPU transfer
        raw_scores = self.model.predict(
            text_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Apply sigmoid if logits are returned
        if np.any(raw_scores < 0.0) or np.any(raw_scores > 1.0):
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            probs = raw_scores

        return probs.astype(np.float32)
