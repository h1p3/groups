"""Bridge between the hypergraph retrieval system (Store/Puller/ContextState)
and the semantic constructor's include-side mechanisms — token-level attract
and sentence-level include-guard (ARCHITECTURE.md §10).

The idea: instead of a user typing out by hand what generation should stay
anchored to, derive it directly from whatever context is *currently pulled
into the active window*. As the conversation grows and ``Puller`` keeps
swapping fresh material in and stale material out (``AutoPullLoop``), the
attract/include target updates automatically along with it — no separate
bookkeeping needed. This is the mechanism for a "dynamic infinite context":
``Store`` can grow without bound, ``Puller`` keeps the *active* window small,
and this module keeps generation anchored to whatever's live in that window
right now, without ever needing the full history in the prompt.

Both functions here re-embed each active node's text through the *same*
embedder a ``VocabIndex``/``SentenceConceptGuard`` was built with
(``vocab_index.embed_query`` / the guard's own ``engine.embed``), rather than
reusing ``Node.embedding`` — that field was very possibly produced by the
generation engine for the Puller's own cosine matching, which is a different
embedding space than a dedicated embedding model's (see the anisotropy
findings in §5.5/§5.1.1); mixing the two would compare vectors that aren't
comparable at all, not just noisily.
"""

from __future__ import annotations

from typing import Iterable

from ..engine.constructor import ConceptSpec


def active_node_texts(store, node_ids: Iterable[str], max_nodes: int | None = None) -> list[str]:
    """Text of the given node ids that still exist in ``store``, in the order
    given. ``max_nodes`` keeps only the *last* ``max_nodes`` of them, so pass
    ``node_ids`` oldest-first (e.g. an ``AutoPullLoop`` recency ``order``
    deque) to keep the most recent ones.

    ``node_ids`` must be an explicitly ordered sequence, not
    ``ContextState.active_ids()`` — that returns a plain ``set`` with no
    recency information at all (``ContextState`` doesn't track insertion
    order; callers that need it, like ``AutoPullLoop.run()``, keep their own
    ``order`` deque alongside it)."""
    ids = list(node_ids)
    if max_nodes is not None:
        ids = ids[-max_nodes:]
    texts = []
    for node_id in ids:
        node = store.get(node_id)
        if node is not None and node.text:
            texts.append(node.text)
    return texts


def context_attract_ids(store, node_ids: Iterable[str], vocab_index, *, max_nodes: int = 4,
                         top_k: int = 20, min_similarity: float = 0.6) -> set[int]:
    """Token IDs pulled toward the vocabulary of the currently active context.

    Re-embeds each of the ``max_nodes`` most recent nodes' text (per
    ``node_ids`` order — see ``active_node_texts``) via
    ``vocab_index.embed_query`` and unions ``VocabIndex.nearest()`` over
    them. Meant to be recomputed after each pull cycle (cheap relative to the
    pull itself: a handful of embed calls, no model generation) so the
    attraction target tracks the active window as it changes.
    """
    attract: set[int] = set()
    for text in active_node_texts(store, node_ids, max_nodes=max_nodes):
        try:
            vec = vocab_index.embed_query(text)
        except Exception:
            continue
        for tid, _text, _sim in vocab_index.nearest(vec, top_k=top_k, min_similarity=min_similarity):
            attract.add(tid)
    return attract


def context_include_spec(store, node_ids: Iterable[str], *, max_nodes: int = 4, max_chars: int = 300,
                         concept: str = "active context") -> ConceptSpec | None:
    """An include-mode ``ConceptSpec`` whose prototypes are the currently
    active context's own text — for use with ``SentenceConceptGuard`` to keep
    generated sentences anchored to what's actually live in the pulled
    context right now, not a fixed topic decided in advance.

    Returns ``None`` when there's no active context yet (nothing to anchor
    to) — callers should skip adding an include concept in that case rather
    than construct one with no prototypes.
    """
    texts = active_node_texts(store, node_ids, max_nodes=max_nodes)
    if not texts:
        return None
    prototypes = [t[:max_chars] for t in texts]
    return ConceptSpec(concept=concept, mode="include", prototypes=prototypes)
