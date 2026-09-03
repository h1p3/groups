from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .base import Group


# Unicode ranges for language detection
_LANG_RANGES: dict[str, list[tuple[int, int]]] = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "ru": [(0x0400, 0x04FF)],
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],
    "de": [(0x00C0, 0x00FF)],
    "fr": [(0x00C0, 0x00FF)],
    "es": [(0x00C0, 0x00FF)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
}


def _char_lang(ch: str) -> str | None:
    cp = ord(ch)
    for lang, ranges in _LANG_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return lang
    return None


class TokenGroup(Group):
    """Абелева группа (Z/2Z)^k для маппинга токенов → элементы группы.

    Группа: (Z/2Z)^k с операцией XOR, единичный элемент — нулевой вектор.
    Маппинг: φ(token_id) = sign(W[token_id] + b) ∈ {0,1}^k.
    """

    name = "token_group"

    def __init__(self, k: int = 64, seed: int = 42):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self._seed = seed
        rng = np.random.RandomState(seed)
        self._W: np.ndarray | None = None
        self._b: np.ndarray = rng.randn(k).astype(np.float32) * 0.1
        # Separate matrix/stream for project_embedding (§12.2 ARCHITECTURE.md):
        # project()/project_per_token() key off the CURRENT logits vector, so
        # a token's element there changes from one generation step to the
        # next -- fine for "fingerprint the current distribution", wrong for
        # a stable concept field. _We/_be depend only on the embedding dim
        # and never change, so the same embedding always lands on the same
        # group element.
        self._We: np.ndarray | None = None
        self._be: np.ndarray | None = None

    def _ensure_W(self, vocab_size: int) -> None:
        if self._W is not None and self._W.shape[0] == vocab_size:
            return
        rng = np.random.RandomState(self._seed)
        self._W = rng.randn(vocab_size, self.k).astype(np.float32) * 0.1

    def _ensure_We(self, embed_dim: int) -> None:
        if self._We is not None and self._We.shape[1] == embed_dim:
            return
        rng = np.random.RandomState(self._seed + 1)  # distinct stream from _W/_b
        self._We = rng.randn(self.k, embed_dim).astype(np.float32) * 0.1
        self._be = rng.randn(self.k).astype(np.float32) * 0.1

    def op(self, a: Any, b: Any) -> Any:
        return np.bitwise_xor(np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32))

    def inverse(self, a: Any) -> Any:
        return np.asarray(a, dtype=np.int32)

    def identity(self) -> Any:
        return np.zeros(self.k, dtype=np.int32)

    def parse(self, text: str) -> Any:
        parts = [int(x) for x in text.strip("[]").split(",")]
        return np.array(parts, dtype=np.int32)

    def to_text(self, a: Any) -> str:
        return str(np.asarray(a, dtype=int).tolist())

    def compact(self, a: Any) -> str:
        v = np.asarray(a, dtype=int)
        return f"[{''.join(str(x) for x in v)}]"

    def project(self, logits: np.ndarray) -> np.ndarray:
        """Маппинг logits → элементы группы: sign(W · logits + b).

        Args:
            logits: shape (vocab_size,) — логиты от модели

        Returns:
            shape (k,) ∈ {0,1}^k — элемент группы
        """
        self._ensure_W(logits.shape[0])
        scores = logits @ self._W + self._b  # (k,)
        return (scores > 0).astype(np.int32)

    def project_per_token(self, logits: np.ndarray) -> np.ndarray:
        """Проекция для каждого токена: shape (vocab_size, k)."""
        self._ensure_W(logits.shape[0])
        return (logits[:, None] * self._W[None, :] + self._b[None, :] > 0).astype(np.int32)

    def project_embedding(self, embedding) -> np.ndarray:
        """Стабильная проекция эмбеддинга в группу: sign(W_e · e + b_e).

        В отличие от `project`/`project_per_token` (зависят от СЫРЫХ logits
        текущего шага генерации — значит нестабильны между шагами, см.
        ARCHITECTURE.md §12.2), эта проекция берёт реальный семантический
        эмбеддинг (например, из `VocabIndex`) и всегда даёт один и тот же
        элемент группы для одного и того же эмбеддинга — то, что нужно для
        стабильного определения поля концепта F_C (`tokens_in_coset`).
        """
        e = np.asarray(embedding, dtype=np.float32).ravel()
        self._ensure_We(e.shape[0])
        scores = self._We @ e + self._be  # (k,)
        return (scores > 0).astype(np.int32)

    def project_embeddings_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Проекция матрицы эмбеддингов (n, embed_dim) → (n, k) элементов
        группы, векторизованно (см. `project_embedding`)."""
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("embeddings must be a 2D array (n, embed_dim)")
        self._ensure_We(arr.shape[1])
        scores = arr @ self._We.T + self._be[None, :]  # (n, k)
        return (scores > 0).astype(np.int32)

    def char_set_to_group_elements(self, chars: str) -> set[tuple[int, ...]]:
        """Множество group elements для набора символов.

        Используется для построения exclude/allow множеств по символам.
        """
        elements = set()
        for ch in chars:
            cp = ord(ch)
            self._ensure_W(65536)
            if cp < self._W.shape[0]:
                elem = (self._W[cp] + self._b > 0).astype(np.int32)
                elements.add(tuple(elem.tolist()))
        return elements

    def token_ids_for_lang(self, lang: str, max_id: int = 65536) -> set[int]:
        """Множество token IDs для языка (по Unicode-диапазонам).

        Примечание: для точного маппинга нужен токенизатор модели.
        Здесь используем Unicode codepoint как приближение.
        """
        ranges = _LANG_RANGES.get(lang, [])
        ids = set()
        for lo, hi in ranges:
            for cp in range(lo, min(hi + 1, max_id)):
                ids.add(cp)
        return ids

    def build_lang_token_ids_from_tokenizer(self, lang: str, vocab_size: int,
                                             tokenize_fn, decode_fn) -> set[int]:
        """Построить точное множество token IDs для языка через токенизатор модели.

        Итерирует ВСЕ token_id в словаре, decode → проверка символов.

        Args:
            lang: код языка ("zh", "ru", "en", ...)
            vocab_size: размер словаря модели
            tokenize_fn: callable(text) → list[int]
            decode_fn: callable(list[int]) → str

        Returns:
            set of token IDs, содержащих символы языка lang
        """
        ranges = _LANG_RANGES.get(lang, [])
        if not ranges:
            return set()
        lang_ids = set()
        for token_id in range(vocab_size):
            try:
                token_str = decode_fn([token_id])
            except Exception:
                continue
            for ch in token_str:
                cp = ord(ch)
                for lo, hi in ranges:
                    if lo <= cp <= hi:
                        lang_ids.add(token_id)
                        break
                else:
                    continue
                break
        return lang_ids

    def lang_to_exclude_set(self, lang: str, vocab_size: int = 65536) -> np.ndarray:
        """Булева маска (vocab_size,) — True для токенов языка lang.

        Использует Unicode codepoints как приближение.
        Для точного маппинга используй build_lang_token_ids_from_tokenizer().
        """
        mask = np.zeros(vocab_size, dtype=bool)
        for token_id in self.token_ids_for_lang(lang, vocab_size):
            if token_id < vocab_size:
                mask[token_id] = True
        return mask

    def build_lang_exclude_mask(self, lang: str, vocab_size: int,
                                tokenize_fn, decode_fn) -> np.ndarray:
        """Булева маска через реальный токенизатор — ТОЧНЫЙ маппинг."""
        lang_ids = self.build_lang_token_ids_from_tokenizer(
            lang, vocab_size, tokenize_fn, decode_fn)
        mask = np.zeros(vocab_size, dtype=bool)
        for tid in lang_ids:
            if tid < vocab_size:
                mask[tid] = True
        return mask

    def load_lang_tokens_from_text(self, text: str, tokenize_fn) -> set[int]:
        """Построить множество token IDs для языка по реальному тексту.

        Args:
            text: текст на целевом языке
            tokenize_fn: callable(text) → list[int] (токенизатор модели)

        Returns:
            set of token IDs, найденных в тексте
        """
        return set(tokenize_fn(text))

    def build_exclude_mask_from_tokens(self, token_ids: set[int], vocab_size: int) -> np.ndarray:
        """Построить булеву маску из множества token IDs."""
        mask = np.zeros(vocab_size, dtype=bool)
        for tid in token_ids:
            if tid < vocab_size:
                mask[tid] = True
        return mask

    def distance(self, a: Any, b: Any) -> int:
        """Hamming distance в (Z/2Z)^k."""
        a_arr = np.asarray(a, dtype=np.int32)
        b_arr = np.asarray(b, dtype=np.int32)
        return int(np.sum(a_arr != b_arr))

    def distance_matrix(self, elements: list[tuple[int, ...]]) -> np.ndarray:
        """Предвычисленная матрица Hamming distances.

        Args:
            elements: список элементов группы (каждый — tuple длины k)

        Returns:
            matrix: shape (n, n) — matrix[i][j] = distance(elements[i], elements[j])
        """
        n = len(elements)
        mat = np.zeros((n, n), dtype=np.int32)
        arr = np.array(elements, dtype=np.int32)  # (n, k)
        for i in range(n):
            for j in range(i + 1, n):
                d = int(np.sum(arr[i] != arr[j]))
                mat[i, j] = d
                mat[j, i] = d
        return mat

    def logit_accumulate(self, logits: np.ndarray) -> np.ndarray:
        """Агрегация логитов через TokenGroup.

        L_accum = Σ softmax(logit[i]) · φ(i) ∈ R^k

        Для метрик: показывает "куда смотрит" модель в групповом пространстве.
        """
        self._ensure_W(logits.shape[0])
        probs = np.exp(logits - np.max(logits))
        probs = probs / (probs.sum() + 1e-10)
        projected = (logits[:, None] * self._W + self._b[None, :] > 0).astype(np.float32)
        accum = (probs[:, None] * projected).sum(axis=0)
        return accum

    def tokens_in_coset(self, center: np.ndarray, elements: np.ndarray,
                        max_distance: int = 3) -> np.ndarray:
        """Найти токены, чьи проекции попадают в косету (max_distance от center).

        Args:
            center: shape (k,) — центр косеты
            elements: shape (vocab_size, k) — проекции всех токенов
            max_distance: максимальная Hamming distance

        Returns:
            mask: shape (vocab_size,) — True для токенов в косете
        """
        diffs = elements != center[None, :]
        dists = diffs.sum(axis=1)
        return dists <= max_distance
