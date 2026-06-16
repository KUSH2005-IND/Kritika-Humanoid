# recognition/identity.py
"""
Identity Decision Engine — converts FAISS similarity scores into identity decisions.

Thresholds (ArcFace R50, cosine similarity):
  > 0.40  : Likely same person — use as RECOGNITION_THRESHOLD
  < 0.30  : Different person
  0.30–0.40 : Uncertain — treat as Unknown

These are empirical; calibrate with your own enrolled data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RECOGNITION_THRESHOLD, UNKNOWN_SAVE_THRESHOLD


class IdentityDecision:
    """
    Converts FAISS similarity score into a human-readable identity decision.
    """

    def __init__(self, threshold=RECOGNITION_THRESHOLD):
        self.threshold = threshold

    def decide(self, name: str, score: float) -> dict:
        """
        Returns:
          {'identity': str, 'score': float, 'is_known': bool, 'confidence': str}
        """
        if score >= self.threshold:
            confidence = "high" if score > 0.55 else "medium"
            return {
                'identity': name,
                'score': score,
                'is_known': True,
                'confidence': confidence
            }
        else:
            return {
                'identity': 'Unknown',
                'score': score,
                'is_known': False,
                'confidence': 'low'
            }
