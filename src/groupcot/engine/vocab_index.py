"""Vocabulary embedding index for the semantic constructor (ARCHITECTURE.md §6.3/V3b).

Builds a cached embedding matrix over a bounded, word-like subset of the model's
vocabulary so ``ConceptConstructor`` can find a concept's semantic *field*
(nearby tokens by cosine distance) instead of only its literal lexicon. This is
what closes the gap V3a leaves open: a lexicon of ``["cat"]`` never blocks
``"cats"`` or ``"kitten"``, but their embeddings sit close to "cat"'s.

Candidates are capped at ``max_candidates`` and taken in token-ID order, which
approximates frequency order for BPE vocabularies (merges are learned most-
frequent-first), so the cap keeps the common vocabulary without an explicit
frequency table. The resulting matrix is cached to disk per (model, vocab
size, cap) so the (relatively expensive) embedding pass runs once per model.

The vocabulary (token IDs) and the embeddings can come from *different*
engines: token IDs must always be the generation model's (that's what
actually gets masked), but a generative decoder's own ``embed()`` produces
low-quality, anisotropic vectors with no real separation between related and
unrelated text (verified empirically — see ARCHITECTURE.md §5.5). Pass a
dedicated embedding model via ``embed_engine`` (e.g. a small e5/bge/LaBSE
GGUF) to get usable similarity while still masking the generation model's own
tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class VocabIndex:
    """Cached {token_id, text} + L2-normalized embedding matrix."""

    def __init__(self, engine, max_candidates: int = 8000,
                 cache_dir: str | Path | None = None, batch_size: int = 64,
                 embed_engine=None, require_word_boundary: bool = True):
        self.engine = engine
        self.embed_engine = embed_engine or engine
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (
            Path.home() / ".cache" / "groupcot" / "vocab_index")
        # Require the RAW (unstripped) surface form to start with a literal
        # leading space -- a BPE word-boundary marker meaning "start of a new
        # word", not a mid-word continuation fragment. Verified empirically
        # (ARCHITECTURE.md §12.7): without this, candidates for morphologically
        # rich languages are dominated by meaningless fragments ("ive", "ord",
        # single Cyrillic letters) that happen to pass isalpha()+len>=2 once
        # stripped -- stripping erases exactly the signal that would have told
        # them apart from genuine whole words. Default True; set False only if
        # you specifically want the old, fragment-inclusive candidate pool.
        self.require_word_boundary = require_word_boundary
        self.token_ids: list[int] = []
        self.texts: list[str] = []
        self.embeddings: np.ndarray | None = None

    @staticmethod
    def _engine_identity(engine) -> str:
        return str(getattr(engine, "model_path", "")
                   or getattr(getattr(engine, "llm", None), "model_path", "")
                   or type(engine).__name__)

    def _cache_key(self) -> str:
        vocab_size = self.engine.vocab_size()
        model_id = self._engine_identity(self.engine)
        embed_id = self._engine_identity(self.embed_engine)
        raw = (f"{model_id}|{vocab_size}|{self.max_candidates}|embed={embed_id}"
               f"|wb={int(self.require_word_boundary)}")
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def embed_query(self, text: str) -> list[float]:
        """Embed an ad-hoc query (e.g. a concept seed) with the same embedder
        used to build this index, so it lives in the same vector space."""
        return self.embed_engine.embed(text)

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self._cache_key()}.npz"

    def _collect_candidates(self) -> list[tuple[int, str]]:
        vocab_size = self.engine.vocab_size()
        candidates: list[tuple[int, str]] = []
        seen: set[str] = set()
        for tid in range(vocab_size):
            if len(candidates) >= self.max_candidates:
                break
            try:
                text = self.engine.detokenize([tid])
            except Exception:
                continue
            if self.require_word_boundary and not text.startswith(" "):
                continue  # mid-word continuation fragment, not a whole word
            word = text.strip()
            if len(word) < 2 or not word.isalpha():
                continue
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((tid, text))
        return candidates

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        embed_batch = getattr(self.embed_engine, "embed_batch", None)
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            if embed_batch is not None:
                vectors.extend(embed_batch(chunk))
            else:
                vectors.extend(self.embed_engine.embed(t) for t in chunk)
        return vectors

    def build(self, force: bool = False) -> None:
        """Load from cache, or embed the candidate vocabulary and cache it."""
        cache_path = self._cache_path()
        if not force and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.token_ids = data["token_ids"].tolist()
            self.texts = data["texts"].tolist()
            self.embeddings = data["embeddings"]
            return

        candidates = self._collect_candidates()
        self.token_ids = [tid for tid, _ in candidates]
        self.texts = [text for _, text in candidates]
        if not self.token_ids:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        vectors = self._embed_texts([text.strip() for text in self.texts])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = arr / norms

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            token_ids=np.array(self.token_ids, dtype=np.int64),
            texts=np.array(self.texts, dtype=object),
            embeddings=self.embeddings,
        )

    def nearest(self, query_vec, top_k: int = 40,
                min_similarity: float = 0.55) -> list[tuple[int, str, float]]:
        """Return ``(token_id, text, cosine_similarity)`` for the nearest tokens."""
        if self.embeddings is None:
            self.build()
        if self.embeddings.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        sims = self.embeddings @ q
        order = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in order:
            sim = float(sims[i])
            if sim < min_similarity:
                break
            results.append((self.token_ids[i], self.texts[i], sim))
        return results
