# Задачи GroupCOT

> Актуальный список задач. Отражает завершённое ядро фильтрации логитов,
> недавние исправления (GUI-старт, CUDA, документация) и новый функционал
> по семантическому конструктору и детерминатору полноты семантического поля
> (спроектирован в `ARCHITECTURE.md` / `GUI_ARCHITECTURE.md`, к реализации не приступали).

Легенда: `[x]` — сделано; `[ ]` — предстоит; `[✑]` — спроектировано, не реализовано.

---

## 1. Ядро фильтрации логитов — ВЫПОЛНЕНО `[x]`

- [x] `TokenGroup` — коммутативная группа (`Cyclic`/`Vector`), метрика, базис полей.
- [x] `LogitFilter` / `FilterRule` (`mode` text/logit, `pipeline` input/output, `action` exclude/allow).
- [x] `build_lang_token_ids` — сканирование токенов языка (CJK, ~25860 zh для Qwen3-VL).
- [x] `ServerEngine` — `logit_bias` через `/v1/completions`, `/tokenize`, `detokenize`, `build_lang_token_ids`.
- [x] `LlamaCppEngine` — `logits_processor`, `logit_bias`, `blocked_ranges`, `embed/tokenize/detokenize/vocab_size`.
- [x] `LogitsProcessorChain` + процессоры: `LanguageRedirect`, `PatternBlock`, `SemanticShift`, `TokenBias`; плагин-система `@register_processor`.
- [x] `RawLogitsAccessor` — захват raw logits, callback, save/load npz.
- [x] CLI: `python -m groupcot {chat,info,benchmark}` (`--backend`, `--exclude-lang`, `--model`, ...).
- [x] Интеграция с `ContextLoop` (`_make_logits_processor`, `_blocked_ranges`).

## 2. КРИТИЧЕСКИЙ БАГ: исключение языка при temperature>0 — ИСПРАВЛЕНО `[x]`

- [x] Симптом: exclude zh работал при temp=0.0, но китайский проскакивал при 0.7/1.0.
- [x] Причина: `logits_processor`/`logit_bias` в llama-cpp-python 0.3.35 применяются к топ-K
      через сломанный `add_custom` сэмплер; + BPE byte-fallback даёт CJK из «мусорных» токенов.
- [x] Решение: runtime посимвольный фильтр `_generate_filtered()` с ручным `eval`+`eval_logits`
      и откатом KV-кэша при появлении заблокированного символа (`DEFAULT_BLOCKED_RANGES`).
- [x] Проверено: temp 0.0/0.7/1.0 → китайского нет; русский промпт → чистый русский ответ.

## 3. GUI — ВЫПОЛНЕНО `[x]`

- [x] Мгновенный старт: `GroupGUI` открывается на `MockEngine`, `LlamaCppEngine` грузится в фоне
      (`_create_default_engine` → `_autoload_llamacpp` → `_apply_autoload` через `after(0)`).
- [x] Переключение/авто-детект/запуск бэкендов (`_connect_engine`, `_auto_detect_server`,
      `_start_server` — CUDA/CPU `llama-server.exe` на порту 8090).
- [x] Чат с исключением языка: `_chat_worker` → `logits_processor` (LlamaCpp) / `logit_bias` (Server)
      + `blocked_ranges`; `_make_llamacpp_logits_processor` / `_make_logit_bias`; бейдж `[фильтр: zh]`.
- [x] Редактор фильтров (`FilterRule`): `_build_filter_panels` + `_FilterDialog`.
- [x] Документация GUI: `GUI_ARCHITECTURE.md`.

## 4. CUDA / GPU — ВЫПОЛНЕНО `[x]`

- [x] Пересборка `llama-cpp-python` 0.3.35 с CUDA 12.6 (`vcvars64` + `nvcc`, target `sm_86`).
- [x] Runtime-DLL (`cudart64_12`/`cublas`/`cublasLt`) скопированы в `llama_cpp\lib\` (работает без PATH).
- [x] Проверено: модель грузится на RTX 3060 CUDA0, слои на GPU.
- [x] `start.bat`: авто-детект CUDA-тулчейна/MSVC, авто-сборка llama-cpp-python с CUDA, GPU-запуск.
- [x] README: разделы «Running on GPU» / «Запуск на GPU» (EN+RU), CPU-гайд сохранён.

## 5. Документация — ВЫПОЛНЕНО `[x]`

- [x] `README.md` — описание, CLI/GUI, deploy, GPU (EN+RU).
- [x] `ARCHITECTURE.md` — управление логитами, 4 варианта конструктора, Вариант 3, детерминатор поля,
      дуальность включения/исключения, роль коммутативных групп.
- [x] `GUI_ARCHITECTURE.md` — архитектура GUI.

---

## 6. Семантический конструктор `[✑]` (фундамент реализован, поле/метр — далее)

Выбран **Вариант 2 (сем. подобласть в группе) + Вариант 1 (жёсткий лексикон)**;
Вариант 3 (self-query) — спроектирован и частично реализован (лёгкая V3a + дуальность).

### 6.1 Базовая маскировка концепта — ВЫПОЛНЕНО `[x]`
- [x] `ConceptSuppress(Set[int])` в `engine/processors.py` — `logits[ids] = -inf`.
- [x] `LlamaCppEngine.generate(..., concept_ids, attract_ids, attract_weight)` — роутинг в
       `_generate_filtered` (ручной путь → работает при любой temperature, см. §temp>0 баг).
- [x] `base.py`/`mock.py`/`server.py` — расширены сигнатуры; server маппит concept/attract → `logit_bias`.
- [x] Защита: `_blocked_ranges` дефолт `()` (не `None`) во избежание падения при чистом concept.

### 6.2 ConceptConstructor (лёгкая версия, V3a) — ВЫПОЛНЕНО `[x]`
- [x] `ConceptSpec{concept, mode, weight, lexicon[], prototypes[], allowed[]}` (`engine/constructor.py`).
- [x] `construct(intent) -> ConceptSpec` — self-query через `engine.chat` (system+user), запасной
       `generate` (мета-промпт). Надёжно работает на Qwen3-VL (chat-шаблон, а не сырой completion).
- [x] `compile(spec) -> Set[int]` — `tokenize` фраз лексикона + варианты (`" word"`, `"Word"`).
- [x] Юнит-тесты `tests/test_concept.py` (Suppress/Attract/Spec/Constructor + интеграция под env-флагом).

### 6.3 ConceptConstructor (полная версия, V3b) — ВЫПОЛНЕНО `[x]` (алгоритм; качество зависит от модели)
- [x] `engine/vocab_index.py::VocabIndex` — индекс эмбеддингов словаря (до `max_candidates`
      словоподобных токенов, дедуп, кэш на диск по hash(model_path, vocab_size, cap)).
- [x] `ConceptConstructor.compile(spec, vocab_index=...)` — `embed(prototypes/lexicon)` → top-k
      ближайших токенов по косинусу из `VocabIndex.nearest()`; `spec.allowed` вычитается.
- [x] CLI: `--semantic-concept` / `--concept-topk` / `--concept-min-sim` / `--concept-vocab-size`.
- [x] Тесты (`tests/test_concept.py`, fake-engine с контролируемым эмбеддинг-пространством):
      build/dedup/cache-roundtrip/nearest, морфологические варианты, `allowed`. 12/12 зелёные.
- [x] Попутно найден и исправлен реальный баг: `llama-cpp-python` 0.3.35
      `create_embedding(list[str])` возвращает испорченные (сдвинутые) векторы для batch-режима
      — `embed_batch` теперь делает по одному вызову на текст.
- [ ] **Известное ограничение (эмпирическое, не баг в этом коде):** на реальной модели
      (Qwen3VL-4B-Instruct) сырые эмбеддинги анизотропны — `cos(cat,dog)≈cos(cat,http)≈0.99` —
      поэтому `nearest()` на этой модели часто тянет случайные сабворд-фрагменты, а не
      «cats»/«kitten». Требует либо отдельной embedding-модели для `embed()`, либо
      whitening/PCA поверх фонового набора. См. ARCHITECTURE.md §5.5.

### 6.4 Дуальность: включение / притяжение (§3.3 ARCHITECTURE) — ВЫПОЛНЕНО `[x]`
- [x] `ConceptSpec.mode ∈ {exclude, include, attract, constrain}` + `weight`.
- [x] `ConceptAttract` в `engine/processors.py` (bias `+= weight`), `ConceptConstrain` — через exclude+attract.
- [x] CLI `--concept` (repeatable) → `_build_concept_ids` → `concept_ids`/`attract_ids` в `generate`.
- [x] `LanguageRedirect` уже поддерживает `boost_mask` (прообраз включения).

### 6.5 Интеграция с CLI и GUI — ВЫПОЛНЕНО `[x]`
- [x] Поле `--concept` в CLI (`chat`/`benchmark`); устранена двойная загрузка движка
       (`_build_processors`/`_build_concept_ids` принимают готовый `engine`).
- [x] GUI: вкладка «Конструктор» (`gui.py::_build_constructor_tab`) — список правил
      (интент + режим exclude/attract + чекбокс V3b), кнопка «Скомпилировать (self-query)»
      запускает `ConceptConstructor.construct`+`compile` в фоновом потоке (не блокирует UI,
      прогресс через очередь `self._q`), список показывает `✓/…` и число токенов на правило.
      Поле для отдельной embedding-модели (V3b, dual-engine — см. §6.3) с кнопкой
      «Подключить», грузится в фоне так же, как основной движок.
- [x] Скомпилированные `concept_ids`/`attract_ids` подключены в `_chat_worker` — передаются в
      `engine.generate(...)` напрямую (работает без веток по типу движка: и `LlamaCppEngine`, и
      `ServerEngine`, и `MockEngine` принимают эти kwargs). Бейдж в чате показывает
      `[concept -N]`/`[concept +N]`, когда правила применены.
- [x] Проверено: headless-инстанс `GroupGUI` + реальный self-query на Qwen3VL через фоновый
      воркер (тот же `"forbid the word 'cat'"` → 4 токена, как в §6.1) — очередь/UI обновляются
      корректно; отдельно проверена передача `concept_ids`/`attract_ids` в `engine.generate` через
      подмену движка на записывающий вызовы.

> Известное ограничение V3a: токенная блокировка дырява (модель уходит в плюральные/капитализированные
> формы и paraphrase). Полное покрытие требует Варианта 2 (сем. подобласть) — см. §6.3 / §7.

---

## 7. Детерминатор полноты семантического поля `[✑]`

- [ ] `SemanticFieldMeter` (новый модуль).
- [ ] `coverage(concept_seed, blocked_ids, radius)` → `field_size, blocked_in_field,
      coverage_pct, lexicon_pct, expanded_pct`.
- [ ] `leakage(logits, concept_seed, blocked_ids)` — динамическая доля вероятности в `F_C\B`.
- [ ] Дуальная метрика inclusion: `adherence = Σ_{t∈F_C} p_t`, inclusion precision/recall.
- [ ] Пороги: `coverage_pct < X` → расширить маску; `leakage_pct > Y` → Фаза 4.
- [ ] Опора на `TokenGroup`/`Puller` (поле F_C = окрестность в групповой метрике).

---

## 8. Уточняющий цикл (Фаза 4 Варианта 3) — ВЫПОЛНЕНО и проверено на реальных моделях `[x]`

- [x] `engine/guarded_generation.py::SentenceConceptGuard` — концепт = целые прототип-предложения
      (не лексикон слов); классификация по cosine к прототипам, `aggregation="mean"|"max"`.
- [x] `generate_guarded()` — генерация чанками → guard.classify() каждой законченной фразы →
      при совпадении: self-query по тексту утечки → `concept_ids` расширяется → перегенерация
      **той же позиции** с более широкой маской (не повтор с тем же распределением). Лимит
      `max_rejections` на весь вызов.
- [x] Тесты `tests/test_guarded_generation.py` (6/6, детерминированный fake-engine): классификация,
      mean vs max агрегация, reject→widen→accept, `gave_up`, passthrough.
- [x] **Dual-engine:** `VocabIndex`/`ConceptConstructor.compile` принимают `embed_engine` — токены
      маскируются с генеративной модели, эмбеддинги считаются отдельной embedding-моделью.
      Тесты `tests/test_concept.py` (dual-engine routing, cache-key различие). См. ниже почему.
- [x] **Проверено на реальных моделях:** Qwen3VL (генерация) + `multilingual-e5-small` q8_0 GGUF
      (эмбеддинги, скачана: `TwinSunsLLC/multilingual-e5-small-gguf`, ~126MB, `models/`, в
      `.gitignore`). `classify()` на реальных RU-фразах при `threshold=0.90, aggregation="mean"`:
      5/6 верно. **Полный цикл reject→widen→regenerate воспроизведён end-to-end на реальных
      моделях**, не только на fake-engine — промпт про отдых у воды дал "...побережье Черного
      моря..." → отклонено → self-query расширил маску на 30 токенов → перегенерация дала
      "...горный озеро..." (без моря/океана/рыбалки). ~41с на CPU (не оптимизировано на скорость —
      каждый чанк эвалюирует `prompt+accepted` заново, KV-кэш между чанками не переиспользуется).
- [x] Первая попытка (Qwen3VL как источник `embed()`) провалилась — ложные совпадения набирали
      больше, чем истинное (та же анизотропия, что в §6.3) — отсюда и появился dual-engine.
      См. ARCHITECTURE.md §5.1.1 за полной историей находки.
- [x] **Перенесено в GUI** (`gui.py`, GUI_ARCHITECTURE.md §7.1.1): подсекция «Guard-концепты (Фаза 4)»
      внутри вкладки «Конструктор» — прототипы вводятся вручную (не self-query, он их ненадёжно
      даёт), `threshold`/`aggregation`/`max_rejections`/`chunk_tokens` настраиваются в UI, чекбокс
      «Включить в чате (строгий режим)» переключает `_chat_worker` на `_generate_with_guard` →
      `generate_guarded`. Языковые фильтры и обычные concept-правила продолжают работать и в
      guard-режиме (форвардятся в каждый внутренний `engine.generate`).
- [x] **По пути найден и исправлен реальный баг:** `generate_guarded` жёстко прокидывал `top_p`/
      `top_k` в каждый `engine.generate(...)` — это параметры только `LlamaCppEngine`, не часть
      контракта `base.Engine`; `MockEngine`/`ServerEngine` падали с `unexpected keyword argument
      'top_p'`. Юнит-тесты этого не ловили (fake-engine с `**kwargs`-заглушкой), всплыло только на
      headless GUI-тесте с `MockEngine`. Исправлено — `top_p`/`top_k` убраны из принудительно
      форвардимых параметров.

---

## 9. Ранее отложенные (из старого tasks.md) `[ ]`

- [ ] Group-level pull scoring (`TokenGroup` внутри узла).
- [ ] Real-time logits visualization в GUI.
- [x] ~~A/B testing: `ServerEngine` vs `LlamaCppEngine` (качество).~~ Снято —
      `ServerEngine` удалён (§14), сравнивать больше не с чем.

---

## 10. Тесты и качество

- [x] 91 тест — все зелёные (ядро фильтрации, процессоры, конструктор, Фаза 4, GUI).
- [ ] Тесты для `ConceptSuppress` / `ConceptConstructor` (V3a) / `SemanticFieldMeter`.
- [ ] Тест на дуальность (exclude/allow) через `LanguageRedirect.boost_mask`.

## 11. Метрики (целевые)

- Coverage исключения: доля поля F_C, попавшая в маску (цель > 99%).
- Adherence включения: доля вероятности в желаемом поле F_C (цель > 95%).
- Leakage: остаточная вероятность в `F_C\B` (цель < 1%).
- Latency: оверхед фильтрации на шаг (цель < 1 мс статически; динамика — по логитам).
- VRAM: < 5 GB для Qwen3-VL-4B на GPU.

## 12. Текущее состояние (сводно)

- [x] Исключение языка работает при ЛЮБОЙ temperature (runtime посимвольный откат).
- [x] GUI открывается мгновенно, движок (LlamaCppEngine, GPU) догружается в фоне.
- [x] CUDA: модель на RTX 3060, все слои на GPU; GPU-режим в `start.bat` + README.
- [x] Документация: `README.md`, `ARCHITECTURE.md`, `GUI_ARCHITECTURE.md`.
- [x] Семантический конструктор (V3a+V3b) и Фаза 4 (в т.ч. include-режим, §13) — реализованы,
      в CLI и GUI. Детерминатор поля (`SemanticFieldMeter`) — всё ещё `[✑]`, не начат.

---

## 13. Include ↔ retrieval: дуальность Фазы 4 и динамический контекст — ВЫПОЛНЕНО `[x]` (механизм; сила attract — открытый вопрос)

Подробный разбор: ARCHITECTURE.md §10. Кратко:

- [x] `GuardViolation` (`NamedTuple`, обратно совместим с `(spec, similarity)`) — `kind="exclude"|"include"`.
- [x] `SentenceConceptGuard.classify()` — дуален по `spec.mode`: exclude (сходство выше threshold —
      отклонить) и include/attract (сходство ниже `include_threshold` до **любого** include-концепта —
      отклонить; несколько include-концептов — OR). Exclude проверяется первым при конфликте.
- [x] `generate_guarded()` — include-виджен принципиально другой: `attract_ids` компилируется один раз
      из спеки (не self-query — цель уже известна), повтор отказа **эскалирует** `attract_weight`
      (не расширяет множество), с потолком `max_attract_weight` (по умолчанию `3×attract_weight`).
- [x] Счётчик `gave_up` — теперь **на позицию** (сброс при каждом принятом предложении), не кумулятивно
      на весь вызов — баг найден собственным тестом на эскалацию до того, как ушёл в прод.
- [x] `context/attractor.py` (новый) — `active_node_texts`/`context_attract_ids`/`context_include_spec`:
      мост от `Store`/`Puller`-контекста к token-level attract и include-guard. Берёт `node_ids` явным
      параметром (не `ContextState.active_ids()` — это `set`, без порядка).
- [x] `AutoPullLoop(..., vocab_index=...)` — опционально пересчитывает `attract_ids` из активного окна
      после каждого pull-цикла, передаёт в `engine.generate(...)`. `None` по умолчанию — старое
      поведение не меняется.
- [x] Тесты: `tests/test_guarded_generation.py` (11/11, +5 новых), `tests/test_attractor.py` (7/7,
      новый файл), `tests/test_loop.py` (5/5, +2 новых).
- [x] **Проверено на реальных моделях** (Qwen3VL + e5, dual-engine) — нашлись две реальные проблемы:
  1. Дефолтный `attract_weight=5.0` был откалиброван неверно на порядки для реального масштаба логитов
     этой модели (`std≈2.9`) — приводил к зацикленному повтору одного токена ("еда еда еда...").
     Добавлен `max_attract_weight` + документирована необходимость калибровки под модель.
  2. **Даже откалиброванный вес не смог развернуть генерацию на контрастную тему** (жёсткий тест:
     заставить ответ про "один совет на день" стать про кулинарию) — 6 отказов подряд, `gave_up=True`.
     Это не баг, а реальный практический предел logit-level attract: хорошо **удерживает** уже вероятное
     направление, плохо **навязывает** тему при сильном естественном тяготении модели в другую сторону.
- [ ] **Открытый вопрос / следующий шаг:** для жёсткого разворота темы, вероятно, нужен prompt-level
      канал (вставлять подтянутый контекст явно в системный промпт), а не только logit-bias. Context-attract
      (§10.2) для retrieval, скорее всего, менее уязвим к этой проблеме (цель обычно и так близка к тому,
      о чём модель уже говорит — удержание, не разворот), но это отдельная эмпирическая проверка, не
      сделана.
- [ ] Эксперимент по §10.4 ARCHITECTURE.md (adherence/drift-rate метрики) — не проведён, только
      спроектирован.
- [ ] GUI: `context_attract_ids`/`context_include_spec` не подключены к вкладке «Запуск» — `AutoPullLoop`
      принимает `vocab_index`, но `gui.py::_run_task`/`_worker` его пока не передают.

---

## 14. Отказ от `server`-бэкенда (`ServerEngine`) — ВЫПОЛНЕНО `[x]`

Подробное обоснование: ARCHITECTURE.md §11. Кратко: `server` умел только
exclude через `logit_bias` (post-hoc, аддитивный dict) — а весь функционал,
построенный после (§6-§13) требует сырых pre-sampling logits и ручного цикла
сэмплинга, которых у HTTP API `llama-server` нет и быть не может. `llamacpp` —
единственный реальный бэкенд теперь; `mock` остаётся для тестов.

- [x] `engine/server.py` (`ServerEngine`) удалён целиком.
- [x] `engine/__init__.py::create_engine` — убрана ветка `"server"`.
- [x] `context/loop.py` — `_build_logit_filters`/`_make_logit_bias` удалены;
      `_make_logits_processor` больше не ветвится по типу движка.
- [x] `cli.py` — `--backend server`, `--base-url` убраны из argparse.
- [x] `gui.py` — радиокнопка «server», поле `base_url`, кнопка «Запустить
      llama-server», `_auto_detect_server`/`_check_server_health`/`_start_server`,
      ветка `ServerEngine` в `_chat_worker`/`_make_logit_bias` — всё удалено.
- [x] `scripts/demo.py` — `--backend server`/`--base-url` убраны.
- [x] Удалены целиком: `test_logits.py` (корень репо, standalone debug под HTTP
      API) и `docs/architecture.md` (устаревший дубликат этого документа).
- [x] `tests/test_filter.py` — `test_server_engine_uses_logit_bias`,
      `test_build_logit_filters`, `test_build_logit_filters_no_logit_rules`
      удалены; `test_chat_worker_applies_logit_filters` переписан на
      `logits_processor` вместо `logit_bias`.
- [x] `pyproject.toml` — зависимость `httpx` убрана (использовалась только
      `ServerEngine` и GUI-детектом сервера).
- [x] `config.yaml`, `README.md` (EN+RU), `GUI_ARCHITECTURE.md` — упоминания
      `server`/`base_url` убраны или явно помечены как «удалено, см. §11».
- [x] `start.bat` — блок запуска `llama-server` убран.
- [x] Тесты: 91/91 зелёных (было 94 — минус 3 удалённых server-теста).
      Прогнаны end-to-end смоук-тесты CLI (`--backend mock info`) и headless
      GUI (инстанцирование + вкладки) после удаления — оба работают чисто.
- [ ] **Не удалено сознательно:** бинарники `tools\llama-server.exe`,
      `tools\ggml-rpc-server.exe`, весь `tools\cuda\` (~1.1GB, включая
      собственноручно собранный `ggml-cuda.dll`) — это данные на диске, не
      код; удаление такого объёма не то, что стоит делать попутно без явного
      запроса. `groups\logit_filter.py::LogitFilter` тоже оставлен — это
      самостоятельный юнит-тестируемый класс, не завязанный именно на
      `ServerEngine`, просто без продакшен-вызывающего кода в `loop.py` теперь.

---

## 15. Подмешивание вероятностей (`SemanticMix`) + `SemanticFieldMeter` под групповую метрику — ВЫПОЛНЕНО `[x]` (механизм; качество поля — открытый вопрос)

Подробности: ARCHITECTURE.md §12 (план §12.1-12.6, реальная проверка §12.7,
открытые вопросы §12.8). Заменяет аддитивный `attract_weight` (нашли в §13:
неограниченный сдвиг, ломается на масштабе конкретной модели) на выпуклую
комбинацию `p_final = (1-α)·p_natural + α·p_concept`.

- [x] `TokenGroup.project_embedding`/`project_embeddings_batch` — стабильная
      проекция эмбеддинга в группу (НЕ `project_per_token`, который проецирует
      текущие logits и потому нестабилен между шагами генерации — находка
      этой сессии, до того как что-то построили поверх него, см. §12.2).
      Тесты: детерминированность, различимость, batch==scalar, отдельная
      матрица от `_W`/`_b`.
- [x] `groups/semantic_field.py::build_concept_field` — F_C как точный coset
      через `VocabIndex.embeddings` + `tokens_in_coset`, не порог по cosine.
      5 тестов (identical-embedding-at-distance-0, monotonic radius, полное
      покрытие при `max_distance=k`, пустые seeds, union нескольких seeds).
- [x] `engine/semantic_field_meter.py::SemanticFieldMeter` — `adherence`
      (динамическая, Σp_t по F_C) и `coverage` (статическая, §6). 7 тестов.
- [x] `LlamaCppEngine.generate(mix_ids=, mix_alpha=, mix_weights=)` →
      `_generate_filtered` → `_mix_and_sample`/`_build_mix_probs` (оба
      `@staticmethod`, тестируются без реальной модели) — только
      `LlamaCppEngine` (то же ограничение, что увело от `server`-бэкенда,
      §11/§12.4, только жёстче: нужен полный `p_natural` до сэмплирования).
      10 тестов, включая стресс-тест на экстремальном масштабе логитов
      (`std~1000`) — не падает, не даёт NaN, остаётся в допустимом диапазоне.
- [x] `generate_guarded` (Фаза 4, §13): include-эскалация переключается на
      `mix_alpha` для `LlamaCppEngine` (детект через `isinstance`), fallback
      на старую эскалацию `attract_weight` для остальных движков (MockEngine
      и фейковые движки в юнит-тестах) — это разрешает риск повторить баг
      top_p/top_k (§13): `mix_ids`/`mix_alpha` никогда не улетают в движок,
      который их не поддерживает. 4 новых теста, включая мок-подкласс
      `LlamaCppEngine` (`isinstance` проходит без реальной модели).
- [x] **Проверено на реальной модели** (Qwen3VL + e5) — см. §12.7 за полным
      разбором. Механизм (формула подмешивания) подтверждён изолированными
      тестами; на реальной модели всплыла **отдельная** проблема — качество
      поля `mix_ids`, не формула:
  - [x] Найдено и исправлено: V3a дробит многосложные русские слова на BPE
        фрагменты (33 «сырых» ID из 5 слов, в основном однобуквенные) —
        подмешивание фрагментов на произвольной позиции ломает синтаксис
        сильнее, чем аддитивный сдвиг когда-либо. Фикс —
        `_filter_whole_word_tokens` (leading-space маркер границы слова),
        2 теста.
  - [ ] **Не решено:** даже с чистым полем маленького размера (2 токена)
        подмешивание при `α≥0.3` всё равно скатывается в зацикливание —
        то же самое узкое место, что убило `attract_weight`, просто через
        другой механизм. Подмешивание чинит калибровку веса под модель, не
        чинит «поле из 2 токенов недостаточно для естественной речи».
  - [x] **Комбинированный фикс проверен** (ARCHITECTURE.md §12.8): широкий
        `VocabIndex` (800→6000 кандидатов) + строгий `min_similarity`
        (0.55→0.85) + `VocabIndex.require_word_boundary` (новый параметр,
        `True` по умолчанию — фикс на уровне источника кандидатов, не только
        для подмешивания: старый фильтр `isalpha()+len>=2` применялся
        **после** `strip()` и потому не отличал целое слово от осколка;
        теперь кандидат обязан начинаться с пробела **до** strip). Поле
        выросло со 137 (мусор) до 101 осмысленного слова (`cooking`,
        `breakfast`, `recipe`, `food`, `menu`...), генерация при `α=0.15-0.35`
        показывает настоящий тематический сдвиг без немедленного распада —
        реальный прогресс, не локальный шум. 2 новых теста на
        `require_word_boundary` (дефолт/выключение + cache-key).
  - [ ] **Но не решение — сдвиг компромисса, два новых честных ограничения:**
        (1) кросс-языковое загрязнение — e5 сближает русские прототипы с
        английскими словами Qwen (`cooking` вместо `готовка`), отсюда
        переключение языка внутри ответа; (2) фильтр «только кириллица»
        проверен и **не помогает** — поле схлопывается до 1 токена
        (`рецепт`), зацикливание возвращается сразу на `α=0.15`. Кросс-язык
        и есть источник ширины поля здесь; широкое+чистое-по-языку+без-
        предела-по-α одновременно не достигнуто ничем опробованным.
  - [ ] Высокий `α` (0.5+) всё равно скатывается в зацикливание независимо от
        ширины поля — «безопасный потолок» поднялся с ~0.1 до ~0.35, эффект
        не исчез. Подтверждает: зацикливание — свойство силы давления, не
        конкретно узости поля.
  - [ ] Фразовое (многотокенное) подмешивание — не начато.
  - [ ] `build_concept_field`/`SemanticFieldMeter` (coset-геометрия) не
        прогнаны вместе с подмешиванием на реальной модели — текущая
        проверка использовала `VocabIndex.nearest` (cosine), не coset.
  - [ ] Специфично-языковой широкий пул (больше `max_candidates` **и** отбор
        по языку кандидата, не только по сходству эмбеддинга) — не пробовали.
