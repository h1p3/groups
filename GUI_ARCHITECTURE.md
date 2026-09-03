# GroupCOT — Архитектура GUI

> Отдельный документ по графическому интерфейсу (`src/groupcot/gui.py`,
> класс `GroupGUI`). Дополняет общую `ARCHITECTURE.md` (управление логитами,
> семантический конструктор, детерминатор поля). Весь GUI-функционал
> реализован (`[✓]`), кроме помеченного явно.

---

## 1. Назначение

GUI — это **клиент** подсистемы GroupCOT: визуализирует гипертекст-контекст на
коммутативной группе, управляет движком (mock / llamacpp — `server` был удалён,
см. ARCHITECTURE.md §11), задаёт правила фильтрации (в т.ч. исключение языка)
и ведёт чат с моделью. Построен на `tkinter` (`import tkinter as tk`).

**Точка входа:**
```
python -m groupcot.gui     # → groupcot.gui.main() → GroupGUI() + mainloop()
```
> Важно: `python -m groupcot` запускает **CLI** (`cli.main`), а не GUI.

---

## 2. Жизненный цикл и мгновенный старт `[✓]`

Главная проблема старта — загрузка 4B GGUF-модели блокирует UI-поток. Решение:
GUI открывается **мгновенно** на `MockEngine`, а тяжёлая загрузка
`LlamaCppEngine` идёт в **фоновом потоке**.

Последовательность в `__init__`:
```
GroupGUI.__init__
  ├─ super().__init__(); title/geometry
  ├─ cfg = load_config(...)
  ├─ self._q = queue.Queue()              # канал фон→UI
  ├─ group = Cyclic(64); store; puller; state; prompts; filters=[]
  ├─ self._build_ui()                      # вкладки + status_label
  ├─ self.engine = self._create_default_engine()   # ← мгновенно возвращает MockEngine
  └─ self.after(400,  self._poll)          # цикл опроса очереди
```

- **`_create_default_engine()`** — возвращает `create_engine("mock", embed_dim=8)`
  и планирует `self.after(700, self._autoload_llamacpp)`. Модель **не грузится**
  синхронно ⇒ окно появляется сразу (~0.8 с).
- **`_autoload_llamacpp()`** — в отдельном `threading.Thread` грузит
  `LlamaCppEngine(model_path, n_ctx, n_gpu_layers)`; по готовности
  `self.after(0, lambda: self._apply_autoload(eng, name))`.
- **`_apply_autoload()`** — если движок всё ещё `MockEngine`, подменяет его на
  загруженный `LlamaCppEngine`, выставляет `backend_var="llamacpp"` и
  обновляет статус. Подмена происходит **только в главном потоке** (через
  `after(0)`), что потокобезопасно для tkinter.

Так модель может использовать GPU (см. CUDA-сборку в `ARCHITECTURE.md`/`README`),
не мешая открытию окна.

---

## 3. Структура класса `GroupGUI` (ключевые поля/методы)

| Поле/метод | Назначение |
|------------|------------|
| `self.cfg` | конфиг (`config.yaml`) |
| `self._q` | `queue.Queue` — результаты из фоновых потоков |
| `self.group` / `self.store` / `self.puller` / `self.state` | гипертекст-контекст на группе |
| `self.prompts` / `self.filters` | набор промптов и правила фильтрации (`list[FilterRule]`) |
| `self.engine` | активный движок (Mock/LlamaCpp) |
| `_build_ui()` | notebook + вкладки + `status_label` |
| `_create_default_engine()` / `_autoload_llamacpp()` / `_apply_autoload()` | жизненный цикл движка |
| `_connect_engine()` | переключение бэкенда вручную |
| `_poll()` / `_refresh_state()` | опрос очереди + обновление статуса |
| `_build_chat_tab()` / `_send_chat()` / `_chat_worker()` | чат и фильтрация логитов |
| `_make_llamacpp_logits_processor()` | построение масок из фильтров |
| `_build_filter_panels()` / `_FilterDialog` | редактор правил фильтрации |
| `self.concept_rules` / `self.embed_engine` / `self._vocab_index` | правила семантического конструктора, отдельный embedding-движок (V3b), кэш индекса словаря |
| `_build_constructor_tab()` / `_compile_concepts()` / `_ConceptDialog` | вкладка «Конструктор»: self-query → `ConceptSpec` → `Set[int]` |

---

## 4. Вкладки (Notebook) `[✓]`

`ttk.Notebook` с вкладками, строятся в `_build_ui()`:

1. **Модель** (`_build_model_tab`) — выбор бэкенда (mock/llamacpp), путь GGUF,
   кнопка «Подключить движок», `model_status`.
2. **Группа** (`_build_group_tab`) — параметры группы (`Cyclic`/`Vector`),
   отображение `h` (состояние группы), активные узлы.
3. **Store** (`_build_store_tab`) — узлы гипертекст-контекста, просмотр/поиск.
4. **Run** (`_build_run_tab`) — запуск основного цикла `ContextLoop` (CoT),
   параметры `pull`/`max_steps`, вывод результата.
5. **Конструктор** (`_build_constructor_tab`) — семантический конструктор
   (`ARCHITECTURE.md` §4-6): список концептов (интент + режим exclude/attract +
   опция V3b), поле отдельной embedding-модели, кнопка «Скомпилировать» —
   подробнее в §7.1.
6. **Чат** (`_build_chat_tab`) — диалог с моделью, поле ввода, история.
7. **Фильтры** (`_build_filter_panels` + `_FilterDialog`) — правила фильтрации
   (`FilterRule`), включая исключение языка (сигнал для `blocked_ranges`).

Низ окна: `status_label` («движок: … | узлов … | в контексте … | group …»).

---

## 5. Движок: переключение и запуск `[✓]`

- **MockEngine** — по умолчанию (мгновенный старт, детерминирован).
- **LlamaCppEngine** — подгружается фоном (`_autoload_llamacpp`); либо вручную
  кнопкой «Подключить движок» (`_connect_engine`). Единственный реальный
  бэкенд — `server` (`ServerEngine`) удалён, см. ARCHITECTURE.md §11: он не
  умел ни в `logits_processor`, ни в ручной сэмплинг с откатом KV, ни в то,
  на чём стоит §7.1 ниже (конструктор/guard).

Потокобезопасность: смена `self.engine` всегда в главном потоке (через
`self.after(0, ...)` из фоновых потоков).

---

## 6. Поток чата и фильтрация логитов `[✓]`

```
пользователь → _send_chat()
  ├─ блокирует кнопку «Отправить»
  └─ threading.Thread(target=_chat_worker).start()
                                      │
_chat_worker()  (фоновый поток)       ▼
  ├─ prompt = _build_chat_prompt()
  ├─ если движок — LlamaCppEngine:
  │    logits_processor = _make_llamacpp_logits_processor() (LogitsProcessorChain)
  │    (MockEngine → без фильтрации)
  ├─ если есть фильтры exclude-language (output/logit/language):
  │    blocked_ranges = DEFAULT_BLOCKED_RANGES   # runtime посимвольный фильтр (см. ARCHITECTURE.md §3.1)
  ├─ concept_ids  = self._compiled_concept_ids()   # из вкладки «Конструктор», §7.1
  │  attract_ids = self._compiled_attract_ids()
  ├─ answer = self.engine.generate(prompt, max_tokens=512, temperature=0.7,
  │                                 logits_processor=, blocked_ranges=,
  │                                 concept_ids=, attract_ids=)
  └─ self._q.put(("chat_reply", answer + badge))   # badge: [фильтр: zh] [concept -N] [concept +N]
                                      │
_poll()  (главный поток, every 400ms) ▼
  ├─ drain _q: chat_reply → _append_chat + разблокировать кнопку
  └─ _refresh_state()  (обновление статуса/списков)
```

**Построение масок из фильтров:**
- `_make_llamacpp_logits_processor()` (LlamaCppEngine, единственный реальный
  бэкенд — см. §5 про удаление `server`): для каждого правила
  `output`/`logit`/`language` строит `lang_ids` через
  `TokenGroup.build_lang_token_ids_from_tokenizer(...)`, маску
  `build_exclude_mask_from_tokens` и добавляет `LanguageRedirect(exclude_mask=)`
  (или `boost_mask=` при `action="allow"`). Возвращает `LogitsProcessorChain`.

**Слой фильтрации в чате:**
1. `logits_processor` — предкомпьютенная маска token-ID (язык/концепт),
   применяется pre-sampling внутри `LlamaCppEngine`.
2. `blocked_ranges` — runtime посимвольный фильтр с откатом KV (работает при
   любой temperature, ловит BPE byte-fallback). Это ровно механизм из
   `ARCHITECTURE.md` §3.1.

---

## 7. Система фильтров (`FilterRule`) `[✓]`

Правило (`groupcot.groups.FilterRule`):
- `type` — `language` (и будущие семантические типы);
- `mode` — `logit` (маскировка логитов);
- `pipeline` — `input` / `output` (на каком этапе применять);
- `action` — `exclude` / `allow` (дуальность из `ARCHITECTURE.md` §3.3);
- `value` — напр. `zh`;
- `group_dim`, `enabled`, `id`.

Жизненный цикл:
- `_build_filter_panels()` отрисовывает список и кнопку «добавить»;
- `_FilterDialog` (наследник `tk.Toplevel`) собирает поля и возвращает
  `FilterRule` (`dlg.result`);
- правила хранятся в `self.filters`; применяются в `_chat_worker` и в Run-цикле
  (`filters=self.filters` передаётся в `ContextLoop`).

> Семантический конструктор (§7.1 ниже) реализован **отдельно** от `FilterRule` —
> не как ещё один `type` в диалоге фильтров, а как собственная вкладка со своим
> состоянием (`self.concept_rules`). Причина: `FilterRule` — плоское правило
> «тип/действие/значение», а концепт — двухфазный объект (интент → self-query →
> `ConceptSpec` → `Set[int]`), с отдельным async-шагом компиляции и опциональным
> вторым движком (embedding-модель для V3b). Продавливать это в форму
> `_FilterDialog` было бы искусственным упрощением.

---

## 7.1 Семантический конструктор в GUI (вкладка «Конструктор») `[✓]`

Реализует `ARCHITECTURE.md` §4-6 (`ConceptConstructor`/`ConceptSpec`/`VocabIndex`)
как отдельную вкладку, не завязанную на `FilterRule`.

**Состояние (`_build_constructor_tab`):**
- `self.concept_rules: list[dict]` — каждый элемент:
  `{"intent", "mode" ("exclude"|"attract"), "semantic" (bool, V3b), "spec"
  (ConceptSpec|None), "ids" (Set[int]|None — None пока не скомпилирован)}`.
- `self.embed_engine` — отдельный движок для эмбеддингов (dual-engine, см.
  `ARCHITECTURE.md` §5.1.1/§8): на generative-модели сырые эмбеддинги
  анизотропны и не разделяют смыслы (эмпирически подтверждено в этой сессии),
  поэтому для V3b рекомендуется небольшая e5/bge/LaBSE GGUF, подключаемая
  отдельно от основной модели чата — своя строка пути + кнопка «Подключить»,
  грузится в фоне тем же паттерном, что `_autoload_llamacpp`.
- `self._vocab_index` — кэш `VocabIndex`; инвалидируется, когда меняется
  `self.engine` или `self.embed_engine` (сравнение по идентичности объектов
  в `_compile_concepts_worker`), иначе переиспользуется между компиляциями.

**Добавление концепта:** `_add_concept_rule` → `_ConceptDialog` (аналог
`_FilterDialog`) — поля: интент (multiline), режим (`exclude`/`attract`),
чекбокс «семантическое расширение (V3b)». Результат добавляется в
`self.concept_rules`, список обновляется `_refresh_concept_list()`.

**Компиляция (`_compile_concepts` → `_compile_concepts_worker`, фоновый поток):**
```
для каждого правила:
  ConceptConstructor(self.engine).construct(intent, mode)   # self-query
     → ConceptSpec{lexicon, prototypes, ...}
  если semantic и хотя бы одно правило это просит:
     VocabIndex(self.engine, embed_engine=self.embed_engine).build()  # раз, кэш
  .compile(spec, vocab_index=...)  → Set[int]   # V3a лексикон (+V3b расширение)
  rule["ids"] = Set[int]
  self._q.put(("concept_progress", "..."))   # UI видит прогресс построчно
self._q.put(("concept_done", None))
```
Как и вся остальная асинхронность в этом GUI (§8) — компиляция идёт в фоне,
`_poll()` разбирает `concept_progress`/`concept_done`/`concept_error` из
`self._q` и обновляет виджеты только в главном потоке.

**Применение в чате:** `_compiled_concept_ids()`/`_compiled_attract_ids()`
объединяют `ids` всех скомпилированных правил по режиму (`exclude`/`constrain`
→ concept_ids, `include`/`attract` → attract_ids) и передаются в
`engine.generate(...)` из `_chat_worker` — `LlamaCppEngine` и `MockEngine`
оба принимают эти kwargs напрямую (`ARCHITECTURE.md` §6.1/§6.4), без веток по
типу движка.

**Проверено:** headless-инстанс `GroupGUI` + реальный self-query на Qwen3VL
через фоновый воркер (тот же `"forbid the word 'cat'"`, что и в CLI/тестах —
4 токена); отдельно — передача `concept_ids`/`attract_ids` в `engine.generate`
подменой движка на записывающий вызовы.

### 7.1.1 Фаза 4 в GUI (`SentenceConceptGuard`/`generate_guarded`) `[✓]`

Отдельная подсекция «Guard-концепты (Фаза 4)» внутри вкладки «Конструктор»
(`_build_guard_section`), не смешана с обычными concept-правилами — у неё
принципиально другая форма данных и другой путь генерации.

**Почему отдельно от обычных концептов:** guard'у (`SentenceConceptGuard`)
нужны целые предложения-прототипы (`spec.prototypes`), а self-query по
умолчанию их не даёт надёжно — на практике возвращает `lexicon` и пустой
`prototypes` (проверено эмпирически на реальной модели в этой сессии).
Поэтому у guard-концептов **нет** self-query шага: `_ConceptGuardDialog`
просто берёт название + прототипы построчно от пользователя напрямую в
`ConceptSpec(concept=name, mode="exclude", prototypes=[...])` — никакого
фонового потока не нужно, это мгновенно.

**Состояние:**
- `self.guard_specs: list[ConceptSpec]` — список guard-концептов.
- `self.guard_enabled_var` — чекбокс «Включить в чате (строгий режим)»;
  без него `_chat_worker` идёт по обычному пути, даже если `guard_specs`
  не пуст (Фаза 4 существенно медленнее — несколько вызовов модели на
  предложение, — включать по необходимости).
- `threshold` / `aggregation` (`mean`/`max`) / `max_rejections` /
  `chunk_tokens` — настройки `SentenceConceptGuard`/`generate_guarded`,
  прямо в UI (спинбоксы/комбобокс), не зашиты в код.

**Путь генерации (`_chat_worker` → `_generate_with_guard`):**
```
если guard_enabled_var и guard_specs непусты:
    embed_src = self.embed_engine или self.engine (fallback с тем же
                предупреждением о качестве, что и в §7.1 для V3b)
    guard = SentenceConceptGuard(embed_src, guard_specs, threshold, aggregation)
    result = generate_guarded(
        self.engine, prompt, guard,
        concept_ids=, attract_ids=,        # из обычных concept-правил, §7.1 — компонуется бесплатно
        vocab_index=self._vocab_index,      # переиспользует V3b индекс для расширения при отказе, если он уже построен
        logits_processor=, blocked_ranges=,  # языковые фильтры продолжают работать и здесь
    )
    answer = result.text
    badge += f" [guard: {N} откл.]" (+ предупреждение, если gave_up)
иначе:
    обычный self.engine.generate(...) как раньше
```
Языковые фильтры (`_make_llamacpp_logits_processor`, `blocked_ranges`) и
обычные concept-правила **продолжают действовать** в guard-режиме —
`generate_guarded` просто форвардит их в каждый внутренний вызов
`engine.generate` через `**generate_kwargs`, три канала фильтрации
(§3.2 ARCHITECTURE.md) остаются независимыми и компонуются.

**Найденный и исправленный по пути баг:** `generate_guarded` изначально жёстко
прокидывал `top_p`/`top_k` в каждый `engine.generate(...)` — это параметры,
которые есть только у `LlamaCppEngine`, а не в контракте `base.Engine`
(`base.py`); `MockEngine` падал с `unexpected keyword argument 'top_p'`.
Всплыло именно на GUI-тесте с `MockEngine` (юнит-тесты `generate_guarded`
использовали fake-engine с
`**kwargs`-заглушкой и эту несовместимость не ловили). Исправлено — `top_p`/
`top_k` убраны из форвардимых по умолчанию параметров; вызывающий код передаёт
их через `**generate_kwargs`, только когда точно знает, что движок —
`LlamaCppEngine`.

**Проверено:** headless `GroupGUI` + `MockEngine` (ловит междвижковую
совместимость, без этого теста баг остался бы незамеченным); отдельно —
реальный dual-engine прогон (`LlamaCppEngine` Qwen3VL + `LlamaCppEngine`
multilingual-e5-small) через `generate_guarded` напрямую — без ошибок,
0.7-90с в зависимости от числа отказов.

---

## 8. Асинхронность и потокобезопасность `[✓]`

- **Фоновые потоки:** загрузка движка (`_autoload_llamacpp`), чат
  (`_chat_worker`), компиляция концептов (`_compile_concepts_worker`).
- **Обратная связь UI ← фон:** только через `self._q` (очередь) + `self.after(0,
  ...)`. `_poll()` (главный поток) разбирает очередь и обновляет виджеты —
  строго в главном потоке, как требует tkinter.
- **Чтение/запись `self.engine`:** смена только в главном потоке
  (`_apply_autoload`/`_connect_engine` через `after`); `self._engine_lock`
  (`threading.Lock`) сериализует все фоновые обращения к `self.engine` (чат,
  компиляция концептов, Run-цикл) — `llama_cpp.Llama` не потокобезопасен, два
  потока внутри одного контекста одновременно приводят к access violation.

---

## 9. Поток данных end-to-end (чат с исключением языка) `[✓]`

```
[GUI: Модель] backend=llamacpp, path=Qwen3VL-4B.gguf
   → _autoload_llamacpp (thread) → LlamaCppEngine (GPU) → _apply_autoload
[GUI: Фильтры] добавлено правило: language=zh, action=exclude, pipeline=output
[GUI: Чат] пользователь вводит вопрос
   → _send_chat → _chat_worker (thread)
        → _make_llamacpp_logits_processor(): LanguageRedirect(zh mask)
        → blocked_ranges = DEFAULT_BLOCKED_RANGES
        → engine.generate(..., logits_processor=chain, blocked_ranges=...)
            ↳ (внутри llamacpp) runtime фильтр + откат KV (ARCHITECTURE.md §3.1)
   → _q.put("chat_reply")
[GUI: _poll] → отображает ответ + бейдж «[фильтр: zh]»
```

Результат: модель понимает вопрос, отвечает (напр. по-английски), не выдавая
запрещённый язык — семантическое поле сохраняется, токены подавлены.

---

## 10. Интеграция с общей архитектурой

| Механизм (ARCHITECTURE.md) | Точка в GUI |
|----------------------------|-------------|
| Исключение языка (§3.1) | `_chat_worker` + `blocked_ranges` + `_make_llamacpp_logits_processor` |
| Дуальность включения/исключения (§3.3) | `action=exclude/allow` → `LanguageRedirect(exclude/boost)`; и отдельно `ConceptSuppress`/`ConceptAttract` через вкладку «Конструктор» (§7.1) |
| Семантический конструктор V3a/V3b (§4-6) | вкладка «Конструктор» (§7.1) → `concept_ids`/`attract_ids` в `_chat_worker` `[✓]` |
| Фаза 4 / `SentenceConceptGuard` (§5.1.1) | подсекция «Guard-концепты» во вкладке «Конструктор» → `_generate_with_guard` `[✓]` |
| Детерминатор поля (§7 ARCHITECTURE.md) | будущая индикация в `status_label`/вкладке фильтров (`[✑]`) |
| Коммутативные группы | `self.group`/`self.state`/`self.puller` (визуализация и Run-цикл) |

---

## 11. Контракты модулей (сводно)

| Модуль/метод | Статус | Ответственность |
|--------------|--------|-----------------|
| `gui.py::GroupGUI` | `[✓]` | каркас окна, вкладки, жизненный цикл |
| `_create_default_engine` / `_autoload_llamacpp` / `_apply_autoload` | `[✓]` | мгновенный старт + фон-загрузка движка |
| `_connect_engine` | `[✓]` | переключение бэкенда вручную |
| `_chat_worker` / `_send_chat` / `_build_chat_prompt` | `[✓]` | чат + фильтрация логитов |
| `_make_llamacpp_logits_processor` | `[✓]` | компиляция фильтров в маску |
| `_build_filter_panels` / `_FilterDialog` | `[✓]` | редактор `FilterRule` |
| `_poll` / `_refresh_state` | `[✓]` | опрос очереди + статус |
| `_build_constructor_tab` / `_compile_concepts(_worker)` / `_ConceptDialog` | `[✓]` | вкладка «Конструктор»: self-query → `ConceptSpec` → `Set[int]` (§7.1) |
| `_compiled_concept_ids` / `_compiled_attract_ids` | `[✓]` | агрегация правил → kwargs для `engine.generate` |
| `_build_guard_section` / `_add_guard_rule` / `_ConceptGuardDialog` | `[✓]` | guard-концепты (прототипы вручную, без self-query), §7.1.1 |
| `_generate_with_guard` | `[✓]` | ветка `_chat_worker` → `generate_guarded` (Фаза 4), §7.1.1 |

---

## 12. Статус и возможные улучшения

**Реализовано `[✓]`:** мгновенный старт GUI; фоновая загрузка LlamaCppEngine
(GPU); переключение бэкенда вручную (mock/llamacpp — `server` удалён, §5,
ARCHITECTURE.md §11); чат с исключением языка через
`logits_processor` + `blocked_ranges`; редактор фильтров; визуализация
группового контекста; вкладка «Конструктор» (§7.1) — self-query → `ConceptSpec`
→ `Set[int]` (V3a + опционально V3b с отдельной embedding-моделью), применяется
в чате как `concept_ids`/`attract_ids`; guard-концепты Фазы 4 (§7.1.1) — целые
предложения-прототипы вручную → `SentenceConceptGuard` → `generate_guarded`
вместо обычного `generate()`, с собственными `threshold`/`aggregation`/
`max_rejections`/`chunk_tokens` в UI и чекбоксом «строгий режим».

**Предлагаемые улучшения `[✑]` (см. ARCHITECTURE.md):**
- Индикация `SemanticFieldMeter` (coverage/leakage/adherence) во вкладке фильтров
  и `status_label` — сам `SemanticFieldMeter` тоже ещё не реализован (`[✑]`,
  ARCHITECTURE.md §7).
- Визуальный предпросмотр итоговой маски (список заблокированных/притянутых
  токенов с расшифровкой через `detokenize`, не только счётчик).
- В чате не отображаются отклонённые Фазой 4 предложения (`result.rejected_sentences`
  доступен, но сейчас в бейдж идёт только счётчик) — можно вывести их в лог/tooltip
  для прозрачности, что именно отфильтровано.
