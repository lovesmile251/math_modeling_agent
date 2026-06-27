"""Backward-compatibility re-export of the social-network paper template.

The implementation has moved to ``tools.paper_templates.social_network``.
This module remains importable so existing callers are not broken.
"""

from __future__ import annotations

from tools.paper_templates.social_network import SocialNetworkPaperTemplate


def build_social_network_paper(workspace, problem_text: str) -> str:
    """Build a complete social-network analysis paper (backward-compatible entry point)."""
    return SocialNetworkPaperTemplate(workspace, problem_text).build()
