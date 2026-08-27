import httpx

from .base import Engine


class ServerEngine(Engine):
    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 api_key: str = "no-key", model: str | None = None,
                 timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def generate(self, prompt, grammar=None, max_tokens: int = 128,
                 temperature: float = 0.7, logits_processor=None,
                 logit_bias: dict[int, float] | None = None,
                 blocked_ranges=None) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
        }
        if self.model:
            payload["model"] = self.model
        if grammar:
            payload["grammar"] = grammar
        if logit_bias:
            payload["logit_bias"] = logit_bias
        resp = self._client.post(f"{self.base_url}/v1/completions", json=payload, headers=self._headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def tokenize(self, text: str) -> list[int]:
        """Получить token IDs через /tokenize endpoint llama-server."""
        resp = self._client.post(
            f"{self.base_url}/tokenize",
            json={"content": text},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["tokens"]

    def detokenize(self, token_ids: list[int]) -> str:
        """Получить строку из token IDs через /detokenize endpoint."""
        resp = self._client.post(
            f"{self.base_url}/detokenize",
            json={"tokens": token_ids},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    def build_lang_token_ids(self, lang: str, vocab_size: int) -> set[int]:
        """Точный маппинг: генерируем ВСЕ символы языка → tokenize → уникальные token IDs."""
        from ..groups.token_group import _LANG_RANGES
        ranges = _LANG_RANGES.get(lang, [])
        if not ranges:
            return set()
        all_chars = ""
        for lo, hi in ranges:
            for cp in range(lo, hi + 1):
                all_chars += chr(cp)
        lang_ids = set()
        chunk_size = 5000
        for i in range(0, len(all_chars), chunk_size):
            chunk = all_chars[i:i + chunk_size]
            try:
                resp = self._client.post(
                    f"{self.base_url}/tokenize",
                    json={"content": chunk},
                    headers=self._headers,
                    timeout=30,
                )
                resp.raise_for_status()
                lang_ids.update(resp.json().get("tokens", []))
            except Exception:
                continue
        return lang_ids

    def embed(self, text) -> list[float]:
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            json={"content": text},
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        vecs = data[0]["embedding"]
        if vecs and isinstance(vecs[0], list):
            n = len(vecs)
            dim = len(vecs[0])
            pooled = [0.0] * dim
            for v in vecs:
                for i, x in enumerate(v):
                    pooled[i] += x
            return [x / n for x in pooled]
        return vecs

    def close(self):
        self._client.close()
