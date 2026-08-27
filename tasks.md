# Задачи: group-level фильтрация логитов/токенов

## Приоритет

### P0 — ядро (Phase 1 — DONE)

- [x] TokenGroup: абелева группа (Z/2Z)^k
- [x] LogitFilter: фильтрация logits через TokenGroup
- [x] FilterRule: mode text/logit
- [x] build_lang_token_ids: tokenize full CJK range -> ~9324 zh tokens

### P1 — интеграция (Phase 1 — DONE)

- [x] ServerEngine: logit_bias через /v1/completions
- [x] ServerEngine: build_lang_token_ids via /tokenize
- [x] ServerEngine: detokenize метод
- [x] LlamaCppEngine: logits_processor параметр (requires install)
- [x] AutoPullLoop: _build_logit_filters + _make_logit_bias
- [x] GUI _chat_worker: logit_bias + badge [фильтр: zh]
- [x] GUI _make_logit_bias: uses build_lang_token_ids

### P2 — Phase 2: llama-cpp-python (DONE)

- [x] **Установка llama-cpp-python (CPU-only, MSVC build)**
  - [x] llama-cpp-python 0.3.35 установлен
  - [x] import llama_cpp OK
  - [x] Model loads: Llama(model_path=..., n_gpu_layers=0, logits_all=True)
  - [x] Completion works: create_completion(prompt=..., max_tokens=...)
  - [x] logit_bias works via create_completion

- [x] **LlamaCppEngine: полная переработка** (engine/llamacpp.py)
  - [x] __init__: logits_all=True
  - [x] generate(): передавать logits_processor в create_completion (+ LogitsProcessorList)
  - [x] generate(): logit_bias параметр
  - [x] generate(): blocked_ranges → runtime посимвольный фильтр (manual sampling)
  - [x] _generate_filtered(): ручной цикл eval + откат при появлении CJK-символа
  - [x] tokenize(): прямой доступ к tokenizer (llm.tokenize)
  - [x] detokenize(): прямой доступ (llm.detokenize)
  - [x] vocab_size(): через llm.n_vocab()
  - [x] embed(): через create_embedding
  - [x] DEFAULT_BLOCKED_RANGES: CJK диапазоны для runtime-фильтра

- [x] **LogitsProcessorChain** (engine/logits_chain.py)
  - [x] __init__: список processors
  - [x] add(processor): добавить processor
  - [x] __call__(input_ids, scores): Apply chain, вернуть modified scores
  - [x] Компатимен с llama-cpp-python logits_processor API

- [x] **LanguageRedirect** (engine/processors.py)
  - [x] __init__(exclude_ids, boost_ids, exclude_mask, boost_mask, boost_strength)
  - [x] __call__(input_ids, scores): -inf для exclude, +strength для boost
  - [x] Mask-based и ID-based операции
  - [x] Не мутирует входной scores

- [x] **PatternBlock** (engine/processors.py)
  - [x] __init__(block_ids, block_mask, block_pattern)
  - [x] from_tokenizer(): regex → scan vocab → blocked_ids
  - [x] __call__(input_ids, scores): -inf для blocked token IDs

- [x] **SemanticShift** (engine/processors.py)
  - [x] __init__(target_element, token_group, boost_strength, max_distance)
  - [x] __call__(input_ids, scores): boost tokens close to target element
  - [x] Векторизованная дистанция через projections

- [x] **TokenBias** (engine/processors.py)
  - [x] __init__(biases, bias_mask)
  - [x] __call__(input_ids, scores): add bias to specific tokens

- [x] **Интеграция с GUI** (gui.py)
  - [x] Radio button: mock/server/llamacpp (auto-detect)
  - [x] _chat_worker: LlamaCppEngine → LogitsProcessorChain
  - [x] _chat_worker: blocked_ranges при exclude-language фильтре
  - [x] _make_llamacpp_logits_processor(): build chain from filters

- [x] **Интеграция с Loop** (context/loop.py)
  - [x] _build_llamacpp_logit_chain(): для LlamaCppEngine
  - [x] _build_logit_filters(): для ServerEngine (текущее поведение)
  - [x] _make_logits_processor(): auto-detect engine type
  - [x] _make_logit_bias(): ServerEngine only
  - [x] _blocked_ranges(): runtime посимвольный фильтр при exclude-language
  - [x] generate() получает blocked_ranges (step + final)

- [x] **Тесты**
  - [x] test_language_redirect_exclude/boost/exclude_and_boost
  - [x] test_pattern_block_ids/mask
  - [x] test_token_bias
  - [x] test_chain (LogitsProcessorChain)
  - [x] test_semantic_shift
  - [x] test_chain_empty
  - [x] test_does_not_mutate_input
  - [x] test_real_model (LlamaCppEngine + chain)

- [x] **КРИТИЧЕСКИЙ БАГ: фильтр языка не работал при temperature > 0 — ИСПРАВЛЕНО**
  - [x] Симптом: exclude zh работал при temp=0.0, но китайский проскакивал при temp=0.7/1.0
  - [x] Причина 1: llama-cpp-python 0.3.35 применяет `logits_processor`/`logit_bias` через
        сломанный `add_custom` сэмплер (комментарий `# NOTE: This is probably broken` в
        исходниках) — модифицирует только топ-K кандидатов, и поведение расходится по temperature
  - [x] Причина 2: предвычисленная маска токенов НЕДОСТАТОЧНА — BPE-декодирование в контексте
        превращает «мусорные» токены (byte-fallback, декодируемые как '�' по отдельности) в
        валидные CJK-символы (напр. токен 62618 → '�이' в одиночку, но '还이' в контексте)
  - [x] Решение: runtime посимвольный фильтр. `generate()` с `blocked_ranges` запускает
        `_generate_filtered()` — ручной цикл `eval` + `eval_logits[-1]` + сэмплинг, где при
        появлении заблокированного символа токен отвергается (-inf) и делается откат KV-кэша
        к точке ДО начала этого символа (чтобы корректно убрать byte-fallback-предшественников)
  - [x] Проверено: temp 0.0/0.7/1.0 → китайского нет; русский промпт → чистый русский ответ
  - [x] Противоречивый промпт (требует китайский + блокируем) → мусорный вывод (ожидаемо)

### P3 — расширения

- [x] **RawLogitsAccessor**: callback для raw logits, save to file (engine/logits_access.py)
  - [x] __call__: passthrough capture, callback, max_history
  - [x] top_k_tokens, save/load npz, reset
- [x] **CLI**: python -m groupcot chat/info/benchmark (cli.py, __main__.py)
  - [x] --backend server|llamacpp|mock
  - [x] --exclude-lang (repeatable, uses LogitsProcessorChain)
  - [x] --model, --base-url, --n-ctx, --n-gpu-layers
  - [x] benchmark -n N: timing test
- [x] **Custom processor plugin system** (engine/processors.py)
  - [x] @register_processor("name") decorator
  - [x] register_processor("name", cls) direct
  - [x] get_processor_class, list_processors
  - [x] build_chain(specs): build chain from list of dicts
  - [x] Built-in: LanguageRedirect, PatternBlock, SemanticShift, TokenBias

### P4 — расширения (остались)

- [ ] Group-level pull scoring (TokenGroup внутри узла)
- [ ] Real-time logits visualization в GUI
- [ ] A/B testing: ServerEngine vs LlamaCppEngine quality
- [ ] CUDA: install nvcc + build from source (n_gpu_layers=99)

## Метрики

- Coverage: доля токенов, прошедших фильтр (целевая > 99%)
- Latency: overhead фильтрации на шаг (целевая < 1ms)
- Accuracy: точность фильтрации языка (целевая > 99%)
- VRAM: потребление GPU (целевая < 5GB для Qwen3-4B)

## Зависимости

- llama-cpp-python >= 0.2.60 (для logits_processor + raw logits)
- CUDA toolkit или pre-built wheel (для GPU)
- numpy (для проекции W·e_i)

## Текущее состояние

- [x] 59 тестов — все зелёные
- [x] ServerEngine + logit_bias работает (25860 zh tokens при tokenizer-scan)
- [x] GUI badge [фильтр: zh]
- [x] LlamaCppEngine 0.3.35: CPU-only, logits_all=True, manual generation + runtime char-filter
- [x] **exclude-language фильтр работает при ЛЮБОЙ temperature (0.0/0.7/1.0) благодаря runtime посимвольному откату**
- [x] LogitsProcessorChain + 4 processors: LanguageRedirect, PatternBlock, SemanticShift, TokenBias
- [x] RawLogitsAccessor: capture + callback + save/load
- [x] CLI: python -m groupcot {chat,info,benchmark} --backend {mock,server,llamacpp} --exclude-lang zh
- [x] Plugin system: @register_processor, build_chain from specs
- [ ] CUDA заблокирована: нет nvcc, нет pre-built Windows wheel
