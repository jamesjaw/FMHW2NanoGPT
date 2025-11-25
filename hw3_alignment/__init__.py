"""
HW3 Alignment Module
This module contains implementations for NanoGPT alignment using DPO.
"""

from .preference_heuristic import score_text, create_preference_pair

__all__ = ['score_text', 'create_preference_pair']
