# GroupCOT

> Auto-pulling hypertext context on commutative groups for Chain-of-Thought (CoT)
> reasoning with quantized LLMs (Qwen3-VL-4B GGUF), GBNF-constrained decoding,
> structured output, a GUI, a chat CLI, and group-level logit filtering.

---

## English

### What is this?

**GroupCOT** is a framework that augments a language model with a *hypertext
context* built from a **commutative group** over text embeddings. Instead of a
flat prompt, related fragments are composed using group operations
(`Cyclic(n)` / `Vector(dim)`), pulled into the active context on demand
(automatic retrieval every N steps), and steered with:

- **Constrained decoding** — GBNF grammars force each CoT step into an
  `element`-string and the final answer into a structured JSON.
- **Group-level logit filtering** — tokens are redirected / masked at the
  logits level so the model stays on track (e.g. exclude a language).
- **Three-channel filtration** of pulled context.

It ships three backends:

| Backend    | Description |
|------------|-------------|
| `mock`     | Deterministic fake engine (no model). Good for UI/flow tests. |
| `llamacpp` | Local GGUF via `llama-cpp-python` (raw logits + manual sampling). |
| `server`   | Remote `llama-server` over HTTP (`/completions` + `/embeddings`). |

### Requirements

- Python **3.10+** (tested on 3.12).
- For `llamacpp`: `llama-cpp-python` (CPU build by default; CUDA build needs a
  CUDA toolchain / prebuilt wheel).
- For `server`: a running `llama-server` (e.g. from the
  [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)).

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -e .[dev]                 # core + tests
pip install -e .[llamacpp]           # + local GGUF backend
```

### Configuration

Edit `config.yaml` (local file, git-ignored):

```yaml
model:
  backend: "mock"                 # mock | llamacpp | server
  path: "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  base_url: "http://127.0.0.1:8090"
  n_ctx: 8192
  n_gpu_layers: 99                # GPU offload layers (llamacpp/server)
  temperature: 0.7

group:
  type: "vector"                  # vector | cyclic
  cyclic_n: 64
  dim: 2560

pull:
  top_k: 2
  threshold: 0.0
  every: 2                        # auto-pull every N steps
  max_active: 16
  max_steps: 6

grammar:
  element: true                   # GBNF on each CoT step
  final_json: true                # GBNF on final JSON

filters: []                       # context filtration rules
```

### Running — CLI

```bash
# Interactive chat with a local GGUF model, excluding Chinese output
python -m groupcot chat --backend llamacpp \
    --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
    --exclude-lang zh

# Chat via a remote llama-server
python -m groupcot chat --backend server --base-url http://127.0.0.1:8090

# Mock engine (no model needed)
python -m groupcot chat --backend mock

# Engine / vocabulary info
python -m groupcot info --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf

# Generation benchmark
python -m groupcot benchmark --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
```

> `python -m groupcot` launches the **CLI** (chat/info/benchmark), not the GUI.

### Running — GUI

```bash
python -m groupcot.gui
```

The window opens instantly (a `MockEngine` by default) and, if a model path is
configured in `config.yaml`, **`LlamaCppEngine` is loaded in a background
thread** and swapped in automatically once ready. Use the *Модель* tab to
connect a backend manually, or start a `llama-server`.

### Deploy — llama-server (recommended for server backend)

For Qwen3-VL (vision) you must also pass the multimodal projector:

```bash
llama-server \
  -m models/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  -c 8192 --port 8090 --embedding --pooling mean -ngl 99
```

- Default port is **8090** (port 8080 is often taken by other services).
- `--embedding --pooling mean` is required for the `ServerEngine` embeddings.
- On Windows you can simply run **`start.bat`**, which downloads the model if
  missing, detects a CUDA vs CPU `llama-server.exe` under `tools\` (or
  `tools\cuda\`), starts it, and launches the GUI.

### Excluding a language

Two mechanisms (both for the `llamacpp` backend):

1. **Token masking** — precomputed `lang` token IDs are redirected via a
   `LogitsProcessor` (`LanguageRedirect`).
2. **Runtime blocked-ranges** — because BPE can fall back to byte tokens that
   still decode to CJK in context, generation is done manually (`llm.eval` +
   `llm.eval_logits`) with a character-level filter
   (`DEFAULT_BLOCKED_RANGES`) and KV-cache rollback. This works at **any
   temperature** (the earlier `logits_processor`-only approach only worked at
   `temperature=0`).

Pass `--exclude-lang zh` (repeatable) to `chat` / `benchmark`, or set it in the
GUI chat tab.

### Tests

```bash
pytest -q          # 59 tests
```

---

## Русский

### Что это?

**GroupCOT** — фреймворк, который дополняет языковую модель *гипертекст-контекстом*,
построенным на **коммутативной группе** над эмбеддингами текста. Вместо плоского
промпта связанные фрагменты компонуются групповыми операциями
(`Cyclic(n)` / `Vector(dim)`), автоматически подтягиваются в активный контекст
(ретривл каждые N шагов) и управляются через:

- **Ограниченную генерацию** — GBNF-грамматики принуждают каждый шаг CoT к
  строке `element`, а финальный ответ — к структурированному JSON.
- **Групповую фильтрацию логитов** — токены перенаправляются / маскируются на
  уровне logits, удерживая модель в нужном русле (например, исключение языка).
- **Трёхканальную фильтрацию** подтянутого контекста.

Доступны три бэкенда:

| Бэкенд    | Описание |
|-----------|----------|
| `mock`    | Детерминированный «фейковый» движок (без модели). Для тестов UI/потока. |
| `llamacpp`| Локальный GGUF через `llama-cpp-python` (raw logits + ручной сэмплинг). |
| `server`  | Удалённый `llama-server` по HTTP (`/completions` + `/embeddings`). |

### Требования

- Python **3.10+** (проверено на 3.12).
- Для `llamacpp`: `llama-cpp-python` (по умолчанию CPU-сборка; CUDA-сборка
  требует CUDA-тулчейн / готовый wheel).
- Для `server`: запущенный `llama-server` (например, из
  [релиза llama.cpp](https://github.com/ggml-org/llama.cpp/releases)).

### Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -e .[dev]                 # ядро + тесты
pip install -e .[llamacpp]           # + локальный GGUF-бэкенд
```

### Конфигурация

Отредактируйте `config.yaml` (локальный файл, в `.gitignore`):

```yaml
model:
  backend: "mock"                 # mock | llamacpp | server
  path: "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  base_url: "http://127.0.0.1:8090"
  n_ctx: 8192
  n_gpu_layers: 99                # слои для offload на GPU (llamacpp/server)
  temperature: 0.7

group:
  type: "vector"                  # vector | cyclic
  cyclic_n: 64
  dim: 2560

pull:
  top_k: 2
  threshold: 0.0
  every: 2                        # автоподтяг каждые N шагов
  max_active: 16
  max_steps: 6

grammar:
  element: true                   # GBNF на каждый шаг CoT
  final_json: true                # GBNF на финальный JSON

filters: []                       # правила фильтрации контекста
```

### Запуск — CLI

```bash
# Интерактивный чат с локальной GGUF-моделью, без китайского на выходе
python -m groupcot chat --backend llamacpp \
    --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
    --exclude-lang zh

# Чат через удалённый llama-server
python -m groupcot chat --backend server --base-url http://127.0.0.1:8090

# Mock-движок (модель не нужна)
python -m groupcot chat --backend mock

# Информация о движке / словаре
python -m groupcot info --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf

# Бенчмарк генерации
python -m groupcot benchmark --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
```

> `python -m groupcot` запускает **CLI** (chat/info/benchmark), а не GUI.

### Запуск — GUI

```bash
python -m groupcot.gui
```

Окно открывается мгновенно (`MockEngine` по умолчанию), а если в `config.yaml`
указан путь к модели, **`LlamaCppEngine` загружается в фоновом потоке** и
автоматически подменяет движок, как только будет готов. На вкладке *Модель*
можно подключить бэкенд вручную или запустить `llama-server`.

### Деплой — llama-server (рекомендуется для server-бэкенда)

Для Qwen3-VL (мультимодальная) нужно также передать проектор:

```bash
llama-server \
  -m models/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  -c 8192 --port 8090 --embedding --pooling mean -ngl 99
```

- Порт по умолчанию — **8090** (8080 часто занят другими сервисами).
- `--embedding --pooling mean` обязательны для эмбеддингов `ServerEngine`.
- В Windows достаточно запустить **`start.bat`**: он скачает модель при
  отсутствии, определит CUDA- или CPU-сборку `llama-server.exe` в `tools\` (или
  `tools\cuda\`), запустит сервер и откроет GUI.

### Исключение языка

Два механизма (оба для бэкенда `llamacpp`):

1. **Маскировка токенов** — предвычисленные `lang` token ID перенаправляются
   через `LogitsProcessor` (`LanguageRedirect`).
2. **Runtime blocked-ranges** — поскольку BPE может упасть на byte-токены,
   которые всё равно декодируются в CJK в контексте, генерация выполняется
   вручную (`llm.eval` + `llm.eval_logits`) с посимвольным фильтром
   (`DEFAULT_BLOCKED_RANGES`) и откатом KV-кэша. Работает при **любой
   temperature** (предыдущий вариант только с `logits_processor` работал лишь
   при `temperature=0`).

Передавайте `--exclude-lang zh` (можно несколько раз) в `chat` / `benchmark`
либо включайте в GUI на вкладке чата.

### Тесты

```bash
pytest -q          # 59 тестов
```
