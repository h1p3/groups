from __future__ import annotations

import numpy as np

from .token_group import TokenGroup


class LogitFilter:
    """Фильтрация logits через TokenGroup.

    На каждом шаге генерации:
    1. Получаем logits от модели (dim = vocab_size)
    2. Для каждого токена i: φ(i) = TokenGroup.project(logits[i])
    3. Если φ(i) ∈ exclude_set → logits[i] = -inf
    4. Если φ(i) ∉ allow_set → logits[i] = -inf (если allow задан)
    """

    def __init__(self, token_group: TokenGroup, vocab_size: int = 150000):
        self.tg = token_group
        self.vocab_size = vocab_size

    def apply(
        self,
        logits: np.ndarray,
        exclude_masks: list[np.ndarray] | None = None,
        allow_masks: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Применить фильтры к logits.

        Args:
            logits: shape (vocab_size,) — исходные логиты
            exclude_masks: список булевых масок (vocab_size,) — True = исключить
            allow_masks: список булевых масок (vocab_size,) — True = разрешить

        Returns:
            masked logits: shape (vocab_size,) с -inf для запрещённых токенов
        """
        masked = logits.copy()

        if exclude_masks:
            combined_exclude = np.zeros(self.vocab_size, dtype=bool)
            for mask in exclude_masks:
                combined_exclude |= mask[:self.vocab_size]
            masked[combined_exclude] = -np.inf

        if allow_masks:
            combined_allow = np.zeros(self.vocab_size, dtype=bool)
            for mask in allow_masks:
                combined_allow |= mask[:self.vocab_size]
            masked[~combined_allow] = -np.inf

        return masked

    def apply_lang_exclude(self, logits: np.ndarray, langs: list[str]) -> np.ndarray:
        """Исключить токены языков."""
        masks = [self.tg.lang_to_exclude_set(lang, self.vocab_size) for lang in langs]
        return self.apply(logits, exclude_masks=masks)

    def apply_lang_allow(self, logits: np.ndarray, langs: list[str]) -> np.ndarray:
        """Разрешить ТОЛЬКО токены указанных языков."""
        masks = [self.tg.lang_to_exclude_set(lang, self.vocab_size) for lang in langs]
        combined = np.zeros(self.vocab_size, dtype=bool)
        for mask in masks:
            combined |= mask
        return self.apply(logits, allow_masks=[combined])

    def apply_pattern_exclude(self, logits: np.ndarray, token_ids: set[int]) -> np.ndarray:
        """Исключить конкретные token IDs."""
        mask = np.zeros(self.vocab_size, dtype=bool)
        for tid in token_ids:
            if tid < self.vocab_size:
                mask[tid] = True
        return self.apply(logits, exclude_masks=[mask])

    def to_logits_processor(self, exclude_masks=None, allow_masks=None):
        """Конвертация в callable для llama-cpp-python logits_processor.

        Возвращает функцию(prompt, scores) → modified_scores.
        """
        def processor(prompt_token_ids, scores):
            return self.apply(scores, exclude_masks=exclude_masks, allow_masks=allow_masks)
        return processor
