# Архитектура: group-level фильтрация логитов/токенов

## Текущая архитектура (Phase 1+2 — завершён)

### Engine layer
- **ServerEngine**: `/v1/completions` + logit_bias (works), `/v1/chat/completions` IGNORES logit_bias
- **LlamaCppEngine**: llama-cpp-python 0.3.35, logits_all=True, logits_processor, tokenize/detokenize
- **MockEngine**: tests

### Logit-level фильтрация

#### ServerEngine (post-hoc via logit_bias)
- build_lang_token_ids(): tokenize ALL CJK chars → ~9324 zh tokens (accurate mapping)
- logit_bias dict: {token_id: -100.0} для exclude
- Post-hoc passes_output_filters() как fallback

#### LlamaCppEngine (pre-sampling via LogitsProcessorChain)
- LogitsProcessorChain → вызывается ПЕРЕД сэмплированием
- 4 процессора: LanguageRedirect, PatternBlock, SemanticShift, TokenBias
- Модель физически не может сгенерировать запрещённые токены

### Проблемы ServerEngine (остаются)
1. Только exclude/allow — нет modification, boost, redirect
2. logit_bias = 9324 entries → llama-server иногда падает с 500
3. Нет доступа к raw logits для анализа
4. Нет logits_processor (сервер игнорирует)

---

## Phase 2: llama-cpp-python — реализован

### Установка
```bash
# CPU-only (текущая установка, Windows, MSVC build)
pip install llama-cpp-python

# CUDA (потребуется nvcc или pre-built wheel)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Архитектура

```
llama-cpp-python Llama(logits_all=True)
       |
       v
model.eval(tokens)  ->  raw scores (vocab_size, float32)
       |
       v
LogitsProcessorChain(input_ids, scores)
  |-- LanguageRedirect    # exclude lang A, boost lang B
  |-- PatternBlock        # block tokens by regex
  |-- SemanticShift       # shift vector in group space
  +-- TokenBias           # arbitrary per-token additive bias
       |
       v
modified scores -> sampling -> token
```

### Компоненты

#### LogitsProcessorChain (`engine/logits_chain.py`)
- `add(processor)` — добавить процессор
- `__call__(input_ids, scores) → scores` — последовательно применить все
- Совместим с llama-cpp-python `LogitsProcessorList`

#### LanguageRedirect (`engine/processors.py`)
```python
# Mask-based (fast, numpy)
LanguageRedirect(exclude_mask=zh_mask, boost_mask=ru_mask, boost_strength=5.0)
# ID-based
LanguageRedirect(exclude_ids={100, 200}, boost_ids={300, 400})
```

#### PatternBlock (`engine/processors.py`)
```python
# From tokenizer scan
PatternBlock.from_tokenizer(r"\b(password|secret)\b", vocab, tokenize, decode)
# Pre-computed
PatternBlock(block_ids={100, 200})
```

#### SemanticShift (`engine/processors.py`)
```python
SemanticShift(target_element=np.array([1,0,1,0,...]), token_group=tg, boost_strength=2.0)
```

#### TokenBias (`engine/processors.py`)
```python
TokenBias(biases={token_id: 3.0, ...})
```

### LlamaCppEngine (`engine/llamacpp.py`)
```python
engine = LlamaCppEngine(model_path="model.gguf", n_ctx=8192, n_gpu_layers=0)
engine.generate(prompt, logits_processor=chain, logit_bias={...})
engine.tokenize("text")  # → list[int]
engine.detokenize([1,2,3])  # → str
engine.vocab_size()  # → int
engine.embed("text")  # → list[float]
```

### Сравнение ServerEngine vs LlamaCppEngine

| | ServerEngine | LlamaCppEngine |
|---|---|---|
| logit_bias | exclude only | full control |
| logits_processor | IGNORED | works (pre-sampling) |
| raw logits | no | yes (logits_all=True) |
| GPU | via llama-server | native CUDA |
| VRAM | separate process | shared (~4GB for Qwen3-4B) |
| tokenize | /tokenize endpoint | llm.tokenize (local) |

### Интеграция

#### GUI (gui.py)
- Radio button: mock/server/llamacpp (auto-detect)
- _chat_worker(): auto-detect engine → build appropriate filter
- _make_llamacpp_logits_processor(): build chain from filters

#### Loop (context/loop.py)
- _build_llamacpp_logit_chain(): для LlamaCppEngine → LogitsProcessorChain
- _build_logit_filters(): для ServerEngine → logit_bias dict
- _make_logits_processor(): auto-detect engine type

### Конфигурация
```yaml
engine:
  type: llamacpp           # server | llamacpp | mock
  model: models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
  n_gpu_layers: 0          # 99 для CUDA
  n_ctx: 8192
```

### Текущее состояние
- llama-cpp-python 0.3.35: CPU-only (MSVC build)
- 25860 zh token IDs через tokenizer scan (vocab_size=151936)
- 43 unit тестов + 11 processor тестов + integration тест — все зелёные
