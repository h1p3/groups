def create_engine(backend: str, **kwargs):
    if backend == "mock":
        from .mock import MockEngine
        return MockEngine(**kwargs)
    if backend == "llamacpp":
        from .llamacpp import LlamaCppEngine
        return LlamaCppEngine(**kwargs)
    if backend == "server":
        from .server import ServerEngine
        return ServerEngine(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")
