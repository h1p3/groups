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

It ships two backends:

| Backend    | Description |
|------------|-------------|
| `mock`     | Deterministic fake engine (no model). Good for UI/flow tests. |
| `llamacpp` | Local GGUF via `llama-cpp-python` (raw logits + manual sampling). |

> A `server` backend (talking to a remote `llama-server` over HTTP) existed
> earlier and was removed. The concept constructor, guard mode, and
> attract/mixing machinery — all still fully working, `llamacpp`-only — need
> raw pre-sampling logits and a manual sampling loop that an HTTP
> `logit_bias` API can't provide; that's *why* `server` had to go, not
> something that went with it. See ARCHITECTURE.md §11 for the full reasoning.

### Requirements

- Python **3.10+** (tested on 3.12).
- `llama-cpp-python` (CPU wheel by default; GPU requires a source build
  against your CUDA toolkit — see *Running on GPU* below). `start.bat`
  performs this automatically when a CUDA toolkit is detected.

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
  backend: "mock"                 # mock | llamacpp
  path: "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  n_ctx: 8192
  n_gpu_layers: 99                # GPU offload layers
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
connect a backend manually.

### Running on GPU (NVIDIA)

The Python `llamacpp` backend can run **entirely on the GPU**. Since
`llama-cpp-python` only ships CPU wheels on PyPI, build it from source against
your CUDA toolkit **once**:

```bash
# 1) Open an MSVC x64 native prompt (provides cl.exe / INCLUDE / LIB)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86

# 2) Force a source build (ignores the prebuilt CPU wheel)
python -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.35

# 3) Copy CUDA runtime DLLs next to the extension (no PATH tweak at runtime)
copy "%CUDA_PATH%\bin\cudart64_12.dll"    .venv\Lib\site-packages\llama_cpp\lib\
copy "%CUDA_PATH%\bin\cublas64_12.dll"    .venv\Lib\site-packages\llama_cpp\lib\
copy "%CUDA_PATH%\bin\cublasLt64_12.dll"  .venv\Lib\site-packages\llama_cpp\lib\
```

After that, **any `llamacpp` run uses the GPU automatically** — set
`n_gpu_layers` in `config.yaml` (e.g. `99`). Verify:

```bash
python -m groupcot info --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
# logs: ggml_cuda_init: found 1 CUDA devices ... layer N assigned to device CUDA0
```

> Do **not** reinstall via `pip install -e .[llamacpp]` afterwards — that would
> replace your CUDA build with the CPU wheel. Just use `pip install -e .`.

The GUI (`python -m groupcot.gui`) autoloads `LlamaCppEngine` and will use the
GPU automatically once the CUDA build above is in place.

On Windows you can simply run **`start.bat`**, which downloads the model if
missing, builds `llama-cpp-python` with CUDA if a toolkit is detected, and
launches the GUI.

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
pytest -q          # 91 tests
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

Доступны два бэкенда:

| Бэкенд    | Описание |
|-----------|----------|
| `mock`    | Детерминированный «фейковый» движок (без модели). Для тестов UI/потока. |
| `llamacpp`| Локальный GGUF через `llama-cpp-python` (raw logits + ручной сэмплинг). |

> Раньше был ещё `server`-бэкенд (удалённый `llama-server` по HTTP) — убран.
> Семантический конструктор, guard-режим и механика attract/подмешивания —
> всё это работает (только на `llamacpp`) и требует сырых logits до
> сэмплирования и ручного цикла сэмплинга, чего HTTP `logit_bias`-API дать не
> может — именно **поэтому** ушёл `server`, а не наоборот. Подробности —
> ARCHITECTURE.md §11.

### Требования

- Python **3.10+** (проверено на 3.12).
- `llama-cpp-python` (на PyPI только CPU-wheel; GPU требует сборки из
  исходников под ваш CUDA-тулчейн — см. *Запуск на GPU* ниже). `start.bat`
  делает это автоматически при наличии CUDA-тулчейна.

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
  backend: "mock"                 # mock | llamacpp
  path: "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  n_ctx: 8192
  n_gpu_layers: 99                # слои для offload на GPU
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
можно подключить бэкенд вручную.

### Запуск на GPU (NVIDIA)

Python-бэкенд `llamacpp` может работать **целиком на видеокарте**. Так как в
PyPI есть только CPU-wheel, нужно один раз собрать `llama-cpp-python` из
исходников под ваш CUDA-тулчейн:

```bash
# 1) Открыть MSVC x64 native prompt (даёт cl.exe / INCLUDE / LIB)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86

# 2) Принудительная сборка из исходников (игнорирует CPU-wheel)
python -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.35

# 3) Скопировать CUDA runtime-DLL рядом с расширением (PATH не нужен в рантайме)
copy "%CUDA_PATH%\bin\cudart64_12.dll"    .venv\Lib\site-packages\llama_cpp\lib\
copy "%CUDA_PATH%\bin\cublas64_12.dll"    .venv\Lib\site-packages\llama_cpp\lib\
copy "%CUDA_PATH%\bin\cublasLt64_12.dll"  .venv\Lib\site-packages\llama_cpp\lib\
```

После этого **любой запуск `llamacpp` использует GPU автоматически** —
выставьте `n_gpu_layers` в `config.yaml` (например, `99`). Проверка:

```bash
python -m groupcot info --backend llamacpp --model models/Qwen3VL-4B-Instruct-Q4_K_M.gguf
# в логах: ggml_cuda_init: found 1 CUDA devices ... layer N assigned to device CUDA0
```

> Не переустанавливайте потом через `pip install -e .[llamacpp]` — это заменит
> CUDA-сборку на CPU-wheel. Используйте просто `pip install -e .`.

GUI (`python -m groupcot.gui`) сам подгружает `LlamaCppEngine` и будет
использовать GPU автоматически, как только выше сделана CUDA-сборка.

В Windows достаточно запустить **`start.bat`**: он скачает модель при
отсутствии, соберёт `llama-cpp-python` с CUDA при обнаруженном тулчейне и
откроет GUI.

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
pytest -q          # 91 тест
```
