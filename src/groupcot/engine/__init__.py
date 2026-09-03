def create_engine(backend: str, **kwargs):
    if backend == "mock":
        from .mock import MockEngine
        return MockEngine(**kwargs)
    if backend == "llamacpp":
        from .llamacpp import LlamaCppEngine
        return LlamaCppEngine(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")
