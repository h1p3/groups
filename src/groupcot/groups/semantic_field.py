"""Concept field F_C as an exact coset in group space (ARCHITECTURE.md §12.2).

This is the "V2 as originally envisioned" version of §6.1: a field defined
by a finite, exactly-countable region in ``TokenGroup``'s (Z/2Z)^k space,
rather than a cosine-similarity threshold (what ``VocabIndex.nearest`` does —
still used elsewhere, e.g. V3b/§5.5, and not being replaced). Both are valid
ways to define a field; this one gives an exact set with a counting measure,
which is what ``SemanticFieldMeter`` and ``SemanticMix`` (§12.1/§12.3) build
on directly — the same F_C for both, not two different notions of "field".
"""

from __future__ import annotations


def build_concept_field(token_group, vocab_index, seeds: list[str],
                        max_distance: int = 8) -> set[int]:
    """Concept field F_C: union, over each seed, of the coset of radius
    ``max_distance`` around that seed's embedding, projected into
    ``token_group`` via ``project_embedding`` (stable — see §12.2 for why
    ``project_per_token`` can't be used for this).

    ``vocab_index`` supplies both the candidate vocabulary (its
    ``token_ids``/``embeddings``, built once and cached, §5.5/§6.3) and the
    seed embedding (``embed_query``) — so this reuses the exact same
    embedding source V3b already uses, just with a different (exact,
    group-based) notion of "close enough" instead of a cosine threshold.
    """
    if vocab_index.embeddings is None or len(vocab_index.token_ids) == 0:
        vocab_index.build()
    if not seeds or vocab_index.embeddings.shape[0] == 0:
        return set()

    elements = token_group.project_embeddings_batch(vocab_index.embeddings)
    field: set[int] = set()
    for seed in seeds:
        if not seed or not seed.strip():
            continue
        try:
            vec = vocab_index.embed_query(seed)
        except Exception:
            continue
        center = token_group.project_embedding(vec)
        mask = token_group.tokens_in_coset(center, elements, max_distance=max_distance)
        field.update(int(tid) for tid, keep in zip(vocab_index.token_ids, mask) if keep)
    return field
