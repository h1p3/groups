"""Coverage/adherence metrics over a concept field F_C (ARCHITECTURE.md §6/§12.3).

F_C is an explicit, finite set of token ids — typically ``build_concept_field``
(exact coset, §12.2) or ``VocabIndex.nearest`` (cosine threshold, §5.5). This
module doesn't care which produced it; it just measures against whatever set
it's given, so ``SemanticMix``'s ``mix_ids`` and this meter's ``field_ids``
can be the *same* set — one definition of the field, not two.
"""

from __future__ import annotations

import numpy as np


class SemanticFieldMeter:
    """Static and dynamic measures of how a mask/mix relates to a concept
    field F_C."""

    @staticmethod
    def adherence(probs, field_ids) -> float:
        """Σ_{t∈F_C} p_t — the probability mass the *current* distribution
        puts in the desired field. Dual of leakage (§6/§3.3): for exclude,
        leakage asks "how much escaped the mask"; for include, adherence
        asks "how much actually lands where we wanted it to"."""
        probs = np.asarray(probs, dtype=np.float64)
        ids = [tid for tid in field_ids if 0 <= tid < len(probs)]
        if not ids:
            return 0.0
        return float(np.sum(probs[ids]))

    @staticmethod
    def coverage(field_ids, masked_ids) -> dict:
        """Static: how much of F_C a given exclude/attract/mix token set
        actually covers.

        Returns ``field_size``, ``masked_in_field`` (|B ∩ F_C|), and
        ``coverage_pct`` = |B ∩ F_C| / |F_C| * 100 — an exact percentage
        since F_C and B are both finite sets, not an approximation.
        """
        field = set(field_ids)
        masked = set(masked_ids)
        if not field:
            return {"field_size": 0, "masked_in_field": 0, "coverage_pct": 0.0}
        inter = field & masked
        return {
            "field_size": len(field),
            "masked_in_field": len(inter),
            "coverage_pct": 100.0 * len(inter) / len(field),
        }
