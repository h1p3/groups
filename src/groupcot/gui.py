import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .config import load_config
from .context import AutoPullLoop, ContextState, Puller, Store
from .engine import create_engine
from .engine.mock import MockEngine
from .groups import Cyclic, FilterRule, VectorAdd
from .prompts import PromptSet

ROOT = Path(__file__).resolve().parents[2]


class GroupGUI(tk.Tk):
    def __init__(self, config_path=None):
        super().__init__()
        self.title("GroupCOT — гипертекст-контекст на группах")
        self.geometry("1220x780")
        self.minsize(980, 620)

        self.cfg = load_config(config_path or (ROOT / "config.yaml"))
        self._q = queue.Queue()
        self._text_counter = 0
        # llama_cpp.Llama is not thread-safe: two threads calling generate()/eval()
        # on the same context concurrently corrupts its internal KV-cache/logits
        # state and crashes with a native access violation (observed in practice --
        # sending a second chat message before the first reply lands). Every
        # background worker that touches self.engine or self.embed_engine must
        # hold this lock for the duration of that access.
        self._engine_lock = threading.Lock()

        self.group = Cyclic(64)
        self.store = Store(group=self.group)
        self.puller = Puller(self.store, top_k=2, threshold=0.0)
        self.state = ContextState(self.group)
        self.prompts = PromptSet()
        self.filters: list[FilterRule] = []

        self._build_ui()
        self.engine = self._create_default_engine()
        self.after(400, self._poll)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb

        self._build_model_tab(nb)
        self._build_group_tab(nb)
        self._build_store_tab(nb)
        self._build_run_tab(nb)
        self._build_constructor_tab(nb)
        self._build_chat_tab(nb)

        self.status_label = ttk.Label(self, relief="sunken", anchor="w")
        self.status_label.pack(fill="x", side="bottom")

    def _build_model_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text="Модель")

        ttk.Label(frame, text="Бэкенд:").pack(anchor="w")
        self.backend_var = tk.StringVar(value=self.cfg["model"]["backend"])
        try:
            from llama_cpp import Llama as _Llama
            _llamacpp_ok = _Llama is not None
        except ImportError:
            _llamacpp_ok = False
        for b in ("mock", "llamacpp"):
            state = "normal" if (b != "llamacpp" or _llamacpp_ok) else "disabled"
            lbl = b if _llamacpp_ok or b != "llamacpp" else f"{b} (не установлен)"
            ttk.Radiobutton(frame, text=lbl, variable=self.backend_var, value=b, state=state).pack(anchor="w", padx=12)

        ttk.Label(frame, text="Путь GGUF (llamacpp):").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        self.model_path_var = tk.StringVar(value=self.cfg["model"]["path"])
        ttk.Entry(row, textvariable=self.model_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self._browse_model).pack(side="left")

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", pady=8)
        ttk.Button(btns, text="Подключить движок", command=self._connect_engine).pack(side="left")

        self.model_status = ttk.Label(frame, text="engine: не подключён")
        self.model_status.pack(anchor="w")

        ttk.Label(
            frame,
            text=(
                "Примечание: Qwen3-VL требует свежий llama.cpp (b6890+). Только llamacpp-бэкенд "
                "(прямой in-process доступ к сырым logits) — remote llama-server больше не "
                "поддерживается, см. ARCHITECTURE.md §11."
            ),
            foreground="#666",
        ).pack(anchor="w", pady=(14, 0))

    def _browse_model(self):
        path = filedialog.askopenfilename(title="GGUF модель", filetypes=[("GGUF", "*.gguf")])
        if path:
            self.model_path_var.set(path)

    def _create_default_engine(self):
        """Создать engine при старте GUI.

        GUI открывается мгновенно (MockEngine), а тяжёлая загрузка LlamaCppEngine
        выполняется в фоновом потоке, чтобы не блокировать открытие окна.
        """
        self.after(700, self._autoload_llamacpp)
        return create_engine("mock", embed_dim=8)

    def _autoload_llamacpp(self):
        """Фоновая загрузка LlamaCppEngine при старте (не блокирует UI)."""
        model_path = self.model_path_var.get().strip() if hasattr(self, "model_path_var") else ""
        if not model_path:
            model_path = self.cfg["model"].get("path", "")
        full = (ROOT / model_path) if model_path and not os.path.isabs(model_path) else Path(model_path)
        if not full.exists():
            return

        def worker():
            try:
                from .engine.llamacpp import LlamaCppEngine
                eng = LlamaCppEngine(
                    model_path=str(full),
                    n_ctx=self.cfg["model"].get("n_ctx", 8192),
                    n_gpu_layers=self.cfg["model"].get("n_gpu_layers", 99),
                )
                self.after(0, lambda: self._apply_autoload(eng, full.name))
            except Exception as exc:
                self.after(0, lambda: self.model_status.config(
                    text=f"engine: ошибка llamacpp ({exc}), используется mock"))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_autoload(self, eng, name):
        if isinstance(self.engine, MockEngine):
            self.engine = eng
            self.backend_var.set("llamacpp")
            self.model_status.config(text=f"engine: llamacpp подключён (авто, {name})")

    def _connect_engine(self):
        backend = self.backend_var.get()
        try:
            if backend == "llamacpp":
                engine = create_engine(
                    "llamacpp",
                    model_path=self.model_path_var.get(),
                    n_ctx=self.cfg["model"]["n_ctx"],
                    n_gpu_layers=self.cfg["model"]["n_gpu_layers"],
                )
            else:
                engine = create_engine("mock", embed_dim=getattr(self.group, "dim", 8))
            self.engine = engine
            self.model_status.config(text=f"engine: {backend} подключён")
        except Exception as exc:
            self.model_status.config(text=f"ошибка подключения: {exc}")

    def _build_group_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text="Группа и контекст")

        top = ttk.LabelFrame(frame, text="Группа")
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Тип:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.group_type_var = tk.StringVar(value="cyclic")
        ttk.Combobox(top, textvariable=self.group_type_var, values=("cyclic", "vector"), state="readonly", width=10).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(top, text="n (cyclic):").grid(row=0, column=2, sticky="e", padx=6)
        self.n_var = tk.IntVar(value=self.cfg["group"]["cyclic_n"])
        ttk.Spinbox(top, from_=2, to=4096, textvariable=self.n_var, width=8).grid(row=0, column=3)
        ttk.Label(top, text="dim (vector):").grid(row=0, column=4, sticky="e", padx=6)
        self.dim_var = tk.IntVar(value=self.cfg["group"]["dim"])
        ttk.Spinbox(top, from_=16, to=8192, textvariable=self.dim_var, width=8).grid(row=0, column=5)
        ttk.Button(top, text="Применить группу", command=self._apply_group).grid(row=0, column=6, padx=10)

        self.group_info = ttk.Label(top, text="")
        self.group_info.grid(row=1, column=0, columnspan=7, sticky="w", padx=6, pady=(0, 6))

        ctx = ttk.LabelFrame(frame, text="Состояние контекста (ContextState)")
        ctx.pack(fill="both", expand=True, padx=6, pady=6)

        info = ttk.Frame(ctx)
        info.pack(fill="x", padx=6, pady=4)
        ttk.Label(info, text="Агрегат h:").pack(side="left")
        self.h_label = ttk.Label(info, text="", foreground="#0a7")
        self.h_label.pack(side="left", padx=6)

        ctrl = ttk.Frame(ctx)
        ctrl.pack(fill="x", padx=6)
        ttk.Button(ctrl, text="Убрать выбранный узел из контекста", command=self._remove_from_context).pack(side="left")
        ttk.Button(ctrl, text="Сбросить контекст", command=self._reset_context).pack(side="left", padx=6)

        listbox_frame = ttk.Frame(ctx)
        listbox_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.active_list = tk.Listbox(listbox_frame, height=6)
        sb = ttk.Scrollbar(listbox_frame, command=self.active_list.yview)
        self.active_list.config(yscrollcommand=sb.set)
        self.active_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        pull = ttk.LabelFrame(frame, text="Параметры автоподтяга")
        pull.pack(fill="x", padx=6, pady=6)
        self.top_k_var = tk.IntVar(value=self.cfg["pull"]["top_k"])
        self.threshold_var = tk.StringVar(value=str(self.cfg["pull"]["threshold"]))
        self.every_var = tk.IntVar(value=self.cfg["pull"]["every"])
        self.max_active_var = tk.IntVar(value=self.cfg["pull"]["max_active"])
        self.max_steps_var = tk.IntVar(value=self.cfg["pull"]["max_steps"])

        def spin(col, label, var, row=0):
            ttk.Label(pull, text=label).grid(row=row, column=col, sticky="e", padx=4)
            ttk.Spinbox(pull, from_=1, to=100, textvariable=var, width=6).grid(row=row, column=col + 1, sticky="w")

        spin(0, "top_k", self.top_k_var)
        spin(2, "pull_every", self.every_var)
        spin(4, "max_active", self.max_active_var)
        spin(6, "max_steps", self.max_steps_var)
        ttk.Label(pull, text="threshold:").grid(row=0, column=8, sticky="e", padx=4)
        ttk.Entry(pull, textvariable=self.threshold_var, width=8).grid(row=0, column=9, sticky="w")

        flt = ttk.LabelFrame(frame, text="Фильтры")
        flt.pack(fill="x", padx=6, pady=6)
        self._build_filter_panels(flt)

    def _build_store_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text="Материалы")

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", padx=6, pady=6)
        ttk.Button(ctrl, text="Добавить файлы…", command=self._add_files).pack(side="left")
        ttk.Button(ctrl, text="Добавить текст…", command=self._add_text).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Удалить узел", command=self._delete_node).pack(side="left")
        ttk.Button(ctrl, text="Сохранить store…", command=self._save_store).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Загрузить store…", command=self._load_store).pack(side="left")

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=6, pady=6)
        self.nodes_list = tk.Listbox(body)
        sb = ttk.Scrollbar(body, command=self.nodes_list.yview)
        self.nodes_list.config(yscrollcommand=sb.set)
        self.nodes_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.nodes_list.bind("<<ListboxSelect>>", self._on_node_select)

        self.node_info = tk.Text(body, height=10, width=52, wrap="word", state="disabled")
        self.node_info.pack(side="left", fill="both", padx=(6, 0))

        ttk.Label(frame, text="Эмбеддинги узлов строятся движком (engine.embed). Для vector-группы вклад в h = эмбеддинг × score.").pack(anchor="w", padx=6, pady=(0, 6))

    def _build_run_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text="Запуск")

        ttk.Label(frame, text="Задача:").pack(anchor="w", padx=6, pady=(6, 0))
        self.task_text = tk.Text(frame, height=6, wrap="word")
        self.task_text.pack(fill="x", padx=6, pady=4)
        self.task_text.insert("1.0", "Объясни связь гомоморфного шифрования и абелевых групп.")

        self.run_btn = ttk.Button(frame, text="Запустить автоподтяг", command=self._run_task)
        self.run_btn.pack(anchor="w", padx=6)

        ttk.Label(frame, text="Результат:").pack(anchor="w", padx=6, pady=(6, 0))
        self.output_text = tk.Text(frame, wrap="word", state="disabled")
        self.output_text.pack(fill="both", expand=True, padx=6, pady=4)

    def _build_constructor_tab(self, nb):
        """Семантический конструктор (ARCHITECTURE.md §4-6): интент на естественном
        языке -> self-query движка -> ConceptSpec -> token IDs (V3a лексикон +
        опционально V3b семантическое расширение), применяется в чате как
        concept_ids/attract_ids."""
        frame = ttk.Frame(nb)
        nb.add(frame, text="Конструктор")
        self.concept_rules: list[dict] = []
        self.embed_engine = None
        self._vocab_index = None

        embed_box = ttk.LabelFrame(
            frame, text="Embedding-модель для семантического расширения (V3b, опционально)")
        embed_box.pack(fill="x", padx=6, pady=6)
        row = ttk.Frame(embed_box)
        row.pack(fill="x", padx=6, pady=4)
        self.embed_model_path_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.embed_model_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self._browse_embed_model).pack(side="left")
        ttk.Button(row, text="Подключить", command=self._connect_embed_engine).pack(side="left", padx=(6, 0))
        self.embed_status = ttk.Label(
            embed_box,
            text=("embedding-движок: не подключён — семантическое расширение (если включено у "
                  "концепта) будет использовать сам движок генерации; на generative-моделях без "
                  "отдельного обучения под эмбеддинги это часто даёт плохое разделение по смыслу "
                  "(см. ARCHITECTURE.md §5.5) — рекомендуется небольшая e5/bge/LaBSE GGUF-модель"),
            foreground="#666", wraplength=900, justify="left",
        )
        self.embed_status.pack(anchor="w", padx=6, pady=(0, 6))

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(ctrl, text="+ Добавить концепт", command=self._add_concept_rule).pack(side="left")
        ttk.Button(ctrl, text="− Удалить", command=self._remove_concept_rule).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Скомпилировать (self-query)", command=self._compile_concepts).pack(side="left", padx=6)

        self.concept_list = tk.Listbox(frame, height=10)
        self.concept_list.pack(fill="both", expand=True, padx=6, pady=6)

        self.concept_status = ttk.Label(frame, text="Правил: 0 | exclude: 0 токенов | attract: 0 токенов")
        self.concept_status.pack(anchor="w", padx=6, pady=(0, 6))

        ttk.Label(
            frame,
            text=("Каждый концепт компилируется через self-query подключённого движка "
                  "(ARCHITECTURE.md §4-5): движок сам строит JSON-спецификацию "
                  "(lexicon/prototypes), лексикон токенизируется (V3a); при включённом "
                  "семантическом расширении дополнительно подтягиваются ближайшие по эмбеддингу "
                  "токены (V3b). Результат применяется в чате как concept_ids (exclude) / "
                  "attract_ids (attract). mock-движок self-query не поддерживает содержательно."),
            foreground="#666", wraplength=900, justify="left",
        ).pack(anchor="w", padx=6, pady=(0, 6))

        self._build_guard_section(frame)

    def _build_guard_section(self, frame):
        """Фаза 4 (ARCHITECTURE.md §5.1.1): guard по целым предложениям, не только
        токенам — отдельный список концептов с прототипами-предложениями (задаются
        руками, не self-query — см. секцию), и отдельный, более медленный режим
        генерации (generate_guarded: reject -> widen mask -> regenerate)."""
        self.guard_specs: list = []  # list[ConceptSpec], full-sentence prototypes
        self.guard_enabled_var = tk.BooleanVar(value=False)
        self.guard_threshold_var = tk.StringVar(value="0.85")
        self.guard_aggregation_var = tk.StringVar(value="mean")
        self.guard_max_rejections_var = tk.IntVar(value=5)
        self.guard_chunk_tokens_var = tk.IntVar(value=20)

        guard_box = ttk.LabelFrame(
            frame, text="Guard-концепты (Фаза 4 — целые предложения, не только токены)")
        guard_box.pack(fill="both", expand=True, padx=6, pady=6)

        ctrl2 = ttk.Frame(guard_box)
        ctrl2.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(ctrl2, text="+ Добавить guard-концепт", command=self._add_guard_rule).pack(side="left")
        ttk.Button(ctrl2, text="− Удалить", command=self._remove_guard_rule).pack(side="left", padx=6)
        ttk.Checkbutton(ctrl2, text="Включить в чате (строгий режим)",
                        variable=self.guard_enabled_var).pack(side="left", padx=12)

        self.guard_list = tk.Listbox(guard_box, height=5)
        self.guard_list.pack(fill="both", expand=True, padx=6, pady=4)

        settings = ttk.Frame(guard_box)
        settings.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(settings, text="threshold:").pack(side="left")
        ttk.Entry(settings, textvariable=self.guard_threshold_var, width=6).pack(side="left", padx=(2, 10))
        ttk.Label(settings, text="агрегация:").pack(side="left")
        ttk.Combobox(settings, textvariable=self.guard_aggregation_var, values=("mean", "max"),
                     state="readonly", width=6).pack(side="left", padx=(2, 10))
        ttk.Label(settings, text="max_rejections:").pack(side="left")
        ttk.Spinbox(settings, from_=1, to=20, textvariable=self.guard_max_rejections_var,
                   width=5).pack(side="left", padx=(2, 10))
        ttk.Label(settings, text="chunk_tokens:").pack(side="left")
        ttk.Spinbox(settings, from_=5, to=100, textvariable=self.guard_chunk_tokens_var,
                   width=5).pack(side="left", padx=(2, 0))

        ttk.Label(
            guard_box,
            text=("Классифицирует КАЖДОЕ сгенерированное предложение по сходству с прототипами "
                  "(не self-query — прототипы вводятся вручную, по одному на строку). При "
                  "совпадении — не повтор с той же маской, а self-query по тексту утечки расширяет "
                  "concept_ids, и предложение перегенерируется. Требует хорошей embedding-модели "
                  "(поле выше) — на самом чат-движке разделение по смыслу часто неинформативно "
                  "(ARCHITECTURE.md §5.1.1). Медленнее обычного чата (несколько вызовов модели на "
                  "предложение) — включайте только когда правда нужно."),
            foreground="#666", wraplength=900, justify="left",
        ).pack(anchor="w", padx=6, pady=(0, 6))

    def _add_guard_rule(self):
        dlg = _ConceptGuardDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self.guard_specs.append(dlg.result)
            self._refresh_guard_list()

    def _remove_guard_rule(self):
        sel = self.guard_list.curselection()
        if not sel:
            return
        del self.guard_specs[sel[0]]
        self._refresh_guard_list()

    def _refresh_guard_list(self):
        self.guard_list.delete(0, "end")
        for spec in self.guard_specs:
            self.guard_list.insert("end", f"{spec.concept!r} — {len(spec.prototypes)} прототипов")

    def _guard_threshold(self) -> float:
        try:
            return float(self.guard_threshold_var.get().replace(",", "."))
        except ValueError:
            return 0.85

    def _browse_embed_model(self):
        path = filedialog.askopenfilename(title="Embedding GGUF модель", filetypes=[("GGUF", "*.gguf")])
        if path:
            self.embed_model_path_var.set(path)

    def _connect_embed_engine(self):
        path = self.embed_model_path_var.get().strip()
        if not path:
            messagebox.showwarning("Не указан путь", "Укажите путь к embedding GGUF модели.")
            return
        full = (ROOT / path) if not os.path.isabs(path) else Path(path)
        if not full.exists():
            messagebox.showwarning("Модель не найдена", f"Файл не существует: {full}")
            return
        self.embed_status.config(text="embedding-движок: загружается…")

        def worker():
            try:
                from .engine.llamacpp import LlamaCppEngine
                eng = LlamaCppEngine(model_path=str(full), n_ctx=512,
                                     n_gpu_layers=self.cfg["model"].get("n_gpu_layers", 99))
                self._q.put(("embed_engine_ready", (eng, full.name)))
            except Exception as exc:
                self._q.put(("embed_engine_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _add_concept_rule(self):
        dlg = _ConceptDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self.concept_rules.append(dlg.result)
            self._refresh_concept_list()

    def _remove_concept_rule(self):
        sel = self.concept_list.curselection()
        if not sel:
            return
        del self.concept_rules[sel[0]]
        self._refresh_concept_list()

    def _refresh_concept_list(self):
        self.concept_list.delete(0, "end")
        total_exclude = 0
        total_attract = 0
        for rule in self.concept_rules:
            n = len(rule.get("ids") or [])
            mode = rule["mode"]
            if mode in ("include", "attract"):
                total_attract += n
            else:
                total_exclude += n
            sem = " [V3b]" if rule.get("semantic") else ""
            status = "✓" if rule.get("ids") is not None else "…"
            self.concept_list.insert("end", f"{status} [{mode}]{sem} {rule['intent']!r} -> {n} токенов")
        self.concept_status.config(
            text=f"Правил: {len(self.concept_rules)} | exclude: {total_exclude} токенов | "
                 f"attract: {total_attract} токенов")

    def _compiled_concept_ids(self) -> set:
        ids: set = set()
        for rule in self.concept_rules:
            if rule["mode"] not in ("include", "attract") and rule.get("ids"):
                ids |= rule["ids"]
        return ids

    def _compiled_attract_ids(self) -> set:
        ids: set = set()
        for rule in self.concept_rules:
            if rule["mode"] in ("include", "attract") and rule.get("ids"):
                ids |= rule["ids"]
        return ids

    def _compile_concepts(self):
        if not self.concept_rules:
            messagebox.showinfo("Нет концептов", "Сначала добавьте хотя бы один концепт.")
            return
        if getattr(self, "_compiling_concepts", False):
            return  # already running -- avoid redundant duplicate self-query calls
        if isinstance(self.engine, MockEngine):
            messagebox.showwarning(
                "mock-движок",
                "Self-query требует реального движка (llamacpp) — mock вернёт заглушку.")
        self._compiling_concepts = True
        threading.Thread(target=self._compile_concepts_worker, daemon=True).start()

    def _compile_concepts_worker(self):
        # Holds self._engine_lock for the whole compile (self-query calls +
        # optional VocabIndex build touch self.engine/self.embed_engine
        # repeatedly) so it can't interleave with a concurrent chat request on
        # the same not-thread-safe llama_cpp.Llama context.
        with self._engine_lock:
            from .engine.constructor import ConceptConstructor
            engine = self.engine
            ctor = ConceptConstructor(engine)
            need_semantic = any(r.get("semantic") for r in self.concept_rules)
            vocab_index = None
            if need_semantic:
                self._q.put(("concept_progress", "Строю/загружаю индекс эмбеддингов словаря (V3b)…"))
                try:
                    from .engine.vocab_index import VocabIndex
                    embed_engine = self.embed_engine  # None -> VocabIndex falls back to `engine` itself
                    vi = self._vocab_index
                    if vi is None or vi.engine is not engine or vi.embed_engine is not (embed_engine or engine):
                        vi = VocabIndex(engine, embed_engine=embed_engine)
                        vi.build()
                        self._vocab_index = vi
                    vocab_index = vi
                except Exception as exc:
                    self._q.put(("concept_progress", f"V3b индекс недоступен ({exc}), использую только V3a"))

            for i, rule in enumerate(self.concept_rules):
                self._q.put((
                    "concept_progress",
                    f"[{i + 1}/{len(self.concept_rules)}] self-query: {rule['intent']!r}…"))
                try:
                    spec = ctor.construct(rule["intent"], mode=rule["mode"])
                    vi_arg = vocab_index if rule.get("semantic") else None
                    ids = ctor.compile(spec, vocab_index=vi_arg)
                    rule["spec"] = spec
                    rule["ids"] = ids
                except Exception as exc:
                    rule["ids"] = set()
                    self._q.put(("concept_error", f"{rule['intent']!r}: {exc}"))
        self._compiling_concepts = False
        self._q.put(("concept_done", None))

    def _apply_group(self):
        gtype = self.group_type_var.get()
        if gtype == "cyclic":
            new_group = Cyclic(int(self.n_var.get()))
        else:
            new_group = VectorAdd(int(self.dim_var.get()))
        if self.engine.__class__.__name__ == "MockEngine" and hasattr(new_group, "dim"):
            self.engine = create_engine("mock", embed_dim=new_group.dim)
        if hasattr(new_group, "dim") and len(self.store.nodes()) > 0:
            old_dim = getattr(self.group, "dim", None)
            if old_dim != new_group.dim:
                if not messagebox.askyesno("Смена размерности", "Эмбеддинги узлов не совпадут с новой dim. Очистить store?"):
                    return
                self.store = Store(group=new_group)
        self.group = new_group
        self.state = ContextState(self.group)
        self.puller = Puller(self.store, top_k=int(self.top_k_var.get()), threshold=self._threshold())
        self.group_info.config(text=f"{self.group.name} ({self.group.__dict__})")
        self.status_label.config(text=f"группа: {self.group.name}; контекст сброшен")

    def _threshold(self):
        try:
            return float(self.threshold_var.get().replace(",", "."))
        except ValueError:
            return 0.0

    def _reset_context(self):
        self.state = ContextState(self.group)

    def _remove_from_context(self):
        sel = self.active_list.curselection()
        if not sel:
            return
        node_id = self.active_list.get(sel[0])
        self.state.remove(node_id)

    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Файлы для контекста")
        for path in paths:
            p = Path(path)
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                messagebox.showwarning("Не прочитан файл", f"{p.name}: {exc}")
                continue
            self._add_node(p.name, text)

    def _add_text(self):
        text = simpledialog.askstring("Добавить фрагмент", "Текст узла:", parent=self)
        if text:
            self._add_node(f"text{self._text_counter}", text)
            self._text_counter += 1

    def _add_node(self, node_id, text):
        embedding = self.engine.embed(text)
        self.store.add(node_id, text, embedding=embedding)

    def _delete_node(self):
        sel = self.nodes_list.curselection()
        if not sel:
            return
        node_id = self.nodes_list.get(sel[0]).split(" — ", 1)[0]
        self.store.remove(node_id)
        self.state.remove(node_id)

    def _save_store(self):
        path = filedialog.asksaveasfilename(title="Сохранить store", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.store.save(path)

    def _load_store(self):
        path = filedialog.askopenfilename(title="Загрузить store", filetypes=[("JSON", "*.json")])
        if path:
            self.store = Store.load(path, self.group)
            self.puller = Puller(self.store, top_k=int(self.top_k_var.get()), threshold=self._threshold())

    def _build_filter_panels(self, frame):
        self._filter_panels = {}
        self._filter_listboxes = {}
        self.filter_mode_var = tk.StringVar(value="text")
        for pipeline, label in [("input", "Вход (контекст)"), ("output", "Выход (генерация)"), ("feedback", "Feedback loop")]:
            flt = ttk.LabelFrame(frame, text=f"Фильтры: {label}")
            flt.pack(fill="x", padx=6, pady=3)
            hdr = ttk.Frame(flt)
            hdr.pack(fill="x", padx=6, pady=2)
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(hdr, text="включён", variable=var).pack(side="left")
            ttk.Button(hdr, text="+", width=3, command=lambda p=pipeline: self._add_filter_rule(p)).pack(side="left", padx=4)
            ttk.Button(hdr, text="−", width=3, command=lambda: self._remove_filter_rule(pipeline)).pack(side="left")
            if pipeline == "output":
                ttk.Separator(hdr, orient="vertical").pack(side="left", padx=6, fill="y")
                ttk.Label(hdr, text="mode:").pack(side="left")
                ttk.Radiobutton(hdr, text="text", variable=self.filter_mode_var, value="text").pack(side="left")
                ttk.Radiobutton(hdr, text="logit", variable=self.filter_mode_var, value="logit").pack(side="left")
            lb = tk.Listbox(flt, height=3)
            lb.pack(fill="x", padx=6, pady=(0, 4))
            self._filter_panels[pipeline] = var
            self._filter_listboxes[pipeline] = lb

    def _add_filter_rule(self, pipeline="input"):
        mode = self.filter_mode_var.get() if pipeline == "output" else "text"
        dlg = _FilterDialog(self, pipeline=pipeline, mode=mode)
        self.wait_window(dlg)
        if dlg.result:
            self.filters.append(dlg.result)
            self._refresh_filters()

    def _remove_filter_rule(self, pipeline):
        lb = self._filter_listboxes[pipeline]
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        pipeline_rules = [r for r in self.filters if r.pipeline == pipeline]
        if idx < len(pipeline_rules):
            rule = pipeline_rules[idx]
            self.filters = [r for r in self.filters if r.id != rule.id]
            self._refresh_filters()

    def _refresh_filters(self):
        for pipeline, lb in self._filter_listboxes.items():
            lb.delete(0, "end")
            for r in self.filters:
                if r.pipeline != pipeline:
                    continue
                status = "✓" if r.enabled else "✗"
                mode_tag = f" [{r.mode}]" if r.pipeline == "output" else ""
                deps = f" ←{','.join(r.depends_on)}" if r.depends_on else ""
                desc = r.description or r.value
                lb.insert("end", f"{status}{mode_tag} [{r.action}] {r.type}: {r.value}{deps} — {desc}")

    def _on_node_select(self, _event=None):
        sel = self.nodes_list.curselection()
        if not sel:
            return
        node_id = self.nodes_list.get(sel[0]).split(" — ", 1)[0]
        node = self.store.get(node_id)
        if node is None:
            return
        embed_dim = len(node.embedding) if node.embedding is not None else 0
        info = (
            f"id: {node.node_id}\n"
            f"edges: {node.edges}\n"
            f"embedding dim: {embed_dim}\n"
            f"meta: {node.meta}\n\n"
            f"text:\n{node.text[:2000]}"
        )
        self.node_info.config(state="normal")
        self.node_info.delete("1.0", "end")
        self.node_info.insert("1.0", info)
        self.node_info.config(state="disabled")

    def _run_task(self):
        task = self.task_text.get("1.0", "end").strip()
        if not task:
            messagebox.showwarning("Пустая задача", "Введите задачу во вкладке «Запуск».")
            return
        params = {
            "top_k": int(self.top_k_var.get()),
            "threshold": self._threshold(),
            "pull_every": int(self.every_var.get()),
            "max_active": int(self.max_active_var.get()),
            "max_steps": int(self.max_steps_var.get()),
        }
        self.run_btn.config(state="disabled")
        self._set_output("работаю…\n")
        threading.Thread(target=self._worker, args=(task, params), daemon=True).start()

    def _worker(self, task, params):
        try:
            # See self._engine_lock note in _chat_worker -- ContextLoop drives
            # self.engine through many generate() calls; must not interleave
            # with a concurrent chat/concept-compile request on the same
            # not-thread-safe llama_cpp.Llama context.
            with self._engine_lock:
                puller = Puller(self.store, top_k=params["top_k"], threshold=params["threshold"])
                loop = AutoPullLoop(
                    self.engine, self.store, puller, self.group, prompts=self.prompts,
                    max_steps=params["max_steps"], pull_every=params["pull_every"],
                    max_active=params["max_active"], filters=self.filters,
                )
                result = loop.run(task, state=self.state)
            self._q.put(("result", result))
        except Exception as exc:
            self._q.put(("error", exc))

    def _set_output(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)
        self.output_text.config(state="disabled")

    def _show_result(self, result):
        self.run_btn.config(state="normal")
        parts = ["Ответ:", result["answer"], "", "Шаги CoT:"]
        for step in result["steps"]:
            parts.append(f"  #{step['n']}  element: {step['element']}  note: {step['note']}")
        parts += [
            "",
            "Подтянуто узлов:",
            *[f"  шаг {e['step']} → {e.get('node', e.get('event', ''))} (score={e.get('score', '')})" for e in result["context"]["pulled"]],
            "",
            f"Агрегат h: {result['context']['h'][:400]}",
            f"Активных узлов: {result['context']['active_count']}",
        ]
        self._set_output("\n".join(parts))
        self.status_label.config(text=f"готово, шагов: {len(result['steps'])}, подтянуто: {len(result['context']['pulled'])}")

    def _show_error(self, exc):
        self.run_btn.config(state="normal")
        self._set_output(f"ошибка:\n{exc}")
        self.status_label.config(text=f"ошибка: {exc}")

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "result":
                    self._show_result(payload)
                elif kind == "error":
                    self._show_error(payload)
                elif kind == "chat_reply":
                    self._chat_history.append({"role": "assistant", "content": payload})
                    self._append_chat("assistant", payload)
                    self._chat_send_btn.config(state="normal")
                elif kind == "chat_error":
                    self._append_chat("assistant", f"[error: {payload}]")
                    self._chat_send_btn.config(state="normal")
                elif kind == "embed_engine_ready":
                    eng, name = payload
                    self.embed_engine = eng
                    self._vocab_index = None
                    self.embed_status.config(text=f"embedding-движок: подключён ({name})")
                elif kind == "embed_engine_error":
                    self.embed_status.config(text=f"embedding-движок: ошибка ({payload})")
                elif kind == "concept_progress":
                    self.concept_status.config(text=payload)
                elif kind == "concept_done":
                    self._refresh_concept_list()
                elif kind == "concept_error":
                    messagebox.showwarning("Ошибка конструктора", str(payload))
        except queue.Empty:
            pass
        self._refresh_state()
        self.after(400, self._poll)

    def _refresh_state(self):
        h = self.group.to_text(self.state.h)
        if len(h) > 120:
            h = h[:120] + "…"
        self.h_label.config(text=h)
        self.group_info.config(text=f"{self.group.name} | {self.group.__dict__}")
        active = sorted(self.state.active_ids())
        if list(self.active_list.get(0, "end")) != active:
            self.active_list.delete(0, "end")
            for node_id in active:
                self.active_list.insert("end", node_id)
        nodes = [f"{n.node_id} — {n.text[:60]}" for n in self.store.nodes()]
        if list(self.nodes_list.get(0, "end")) != nodes:
            self.nodes_list.delete(0, "end")
            for item in nodes:
                self.nodes_list.insert("end", item)
        self.status_label.config(
            text=f"движок: {self.engine.__class__.__name__} | узлов в store: {len(self.store.nodes())} | "
            f"в контексте: {len(self.state)} | group: {self.group.name}"
        )



    def _build_chat_tab(self, nb):
        frame = ttk.Frame(nb)
        nb.add(frame, text="Чат")
        self._chat_history = []
        self.chat_display = tk.Text(frame, wrap="word", state="disabled", height=20)
        self.chat_display.pack(fill="both", expand=True, padx=6, pady=6)
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill="x", padx=6, pady=(0, 6))
        self.chat_input = tk.Text(input_frame, height=2, wrap="word")
        self.chat_input.pack(side="left", fill="x", expand=True)
        self.chat_input.bind("<Return>", self._on_chat_enter)
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side="left", padx=(6, 0))
        self._chat_send_btn = ttk.Button(btn_frame, text="Отправить", command=self._send_chat)
        self._chat_send_btn.pack()
        ttk.Button(btn_frame, text="Очистить", command=self._clear_chat).pack(pady=(4, 0))

    def _on_chat_enter(self, event):
        if not (event.state & 0x1):
            self._send_chat()
            return "break"

    def _send_chat(self):
        # Guards against the Enter-key binding (or a fast double-click) firing a
        # second request while one is still in flight -- self.engine isn't
        # thread-safe, and disabling the button alone doesn't block <Return>.
        if str(self._chat_send_btn["state"]) == "disabled":
            return
        text = self.chat_input.get("1.0", "end").strip()
        if not text:
            return
        self.chat_input.delete("1.0", "end")
        self._chat_history.append({"role": "user", "content": text})
        self._append_chat("user", text)
        self._chat_send_btn.config(state="disabled")
        threading.Thread(target=self._chat_worker, daemon=True).start()

    def _chat_worker(self):
        try:
            # self.engine (and self.embed_engine, used inside _generate_with_guard)
            # are llama_cpp.Llama-backed and not thread-safe -- hold the lock for
            # the whole request so a concept-compile or Run-tab task can't
            # interleave native calls on the same context and crash the process.
            with self._engine_lock:
                prompt = self._build_chat_prompt()
                from groupcot.engine.llamacpp import LlamaCppEngine

                logits_processor = None
                blocked_ranges = None

                if isinstance(self.engine, LlamaCppEngine):
                    logits_processor = self._make_llamacpp_logits_processor()

                filtered_langs = [f.value for f in self.filters
                                 if f.enabled and f.mode == "logit" and f.pipeline == "output"
                                 and f.type == "language" and f.action == "exclude"]

                # Runtime character-level filter: a precomputed token mask is not
                # enough because BPE contextual decoding can turn "garbage" tokens
                # into valid CJK characters. We additionally reject any generated
                # token whose decoded text introduces a blocked codepoint.
                if filtered_langs:
                    from groupcot.engine.llamacpp import DEFAULT_BLOCKED_RANGES
                    blocked_ranges = DEFAULT_BLOCKED_RANGES

                # Семантический конструктор (вкладка «Конструктор»): скомпилированные
                # concept_ids/attract_ids передаются в engine.generate(...) напрямую.
                concept_ids = self._compiled_concept_ids() or None
                attract_ids = self._compiled_attract_ids() or None

                badge = ""
                if filtered_langs:
                    badge += f" [фильтр: {', '.join(filtered_langs)}]"
                if concept_ids:
                    badge += f" [concept -{len(concept_ids)}]"
                if attract_ids:
                    badge += f" [concept +{len(attract_ids)}]"

                if self.guard_enabled_var.get() and self.guard_specs:
                    answer, guard_badge = self._generate_with_guard(
                        prompt, concept_ids, attract_ids,
                        logits_processor=logits_processor, blocked_ranges=blocked_ranges)
                    badge += guard_badge
                else:
                    answer = self.engine.generate(prompt, max_tokens=512, temperature=0.7,
                                                  logits_processor=logits_processor,
                                                  blocked_ranges=blocked_ranges,
                                                  concept_ids=concept_ids,
                                                  attract_ids=attract_ids)

            self._q.put(("chat_reply", answer + badge))
        except Exception as exc:
            self._q.put(("chat_error", exc))

    def _generate_with_guard(self, prompt, concept_ids, attract_ids, *,
                             logits_processor, blocked_ranges):
        """Фаза 4 (ARCHITECTURE.md §5.1.1): generate_guarded вместо одного
        engine.generate — генерирует чанками, классифицирует каждое законченное
        предложение против guard_specs, при совпадении расширяет маску через
        self-query на тексте утечки и перегенерирует ту же позицию."""
        from .engine.guarded_generation import SentenceConceptGuard, generate_guarded

        embed_src = self.embed_engine or self.engine
        if not hasattr(embed_src, "embed"):
            raise RuntimeError(
                "Guard-режим требует embed() у движка; подключите embedding-модель выше "
                "или обычный движок с embed()")

        guard = SentenceConceptGuard(
            embed_src, self.guard_specs,
            threshold=self._guard_threshold(), aggregation=self.guard_aggregation_var.get())

        result = generate_guarded(
            self.engine, prompt, guard,
            max_tokens=512, chunk_tokens=int(self.guard_chunk_tokens_var.get()),
            max_rejections=int(self.guard_max_rejections_var.get()),
            concept_ids=concept_ids, attract_ids=attract_ids,
            vocab_index=self._vocab_index, temperature=0.7,
            logits_processor=logits_processor, blocked_ranges=blocked_ranges,
        )

        badge = f" [guard: {len(result.rejected_sentences)} откл.]" if result.rejected_sentences else ""
        if result.gave_up:
            badge += " [ФАЗА4: не удалось обойти концепт — попробуйте max_rejections/threshold]"
        return result.text, badge

    def _make_llamacpp_logits_processor(self):
        """Построить LogitsProcessorChain для LlamaCppEngine."""
        from groupcot.engine.logits_chain import LogitsProcessorChain
        from groupcot.engine.processors import LanguageRedirect
        from groupcot.groups.token_group import TokenGroup

        logit_rules = [f for f in self.filters if f.enabled and f.mode == "logit" and f.pipeline == "output"]
        if not logit_rules:
            return None

        vocab_size = self.engine.vocab_size()
        tg = TokenGroup(k=logit_rules[0].group_dim)
        chain = LogitsProcessorChain()

        for rule in logit_rules:
            if rule.type == "language":
                try:
                    lang_ids = set(tg.build_lang_token_ids_from_tokenizer(
                        rule.value, vocab_size, self.engine.tokenize, self.engine.detokenize))
                except Exception:
                    lang_ids = tg.token_ids_for_lang(rule.value, vocab_size)

                mask = tg.build_exclude_mask_from_tokens(lang_ids, vocab_size)

                if rule.action == "exclude":
                    chain.add(LanguageRedirect(exclude_mask=mask))
                else:
                    chain.add(LanguageRedirect(boost_mask=mask))

        return chain if len(chain) > 0 else None

    def _build_chat_prompt(self):
        T = chr(119) + chr(101) + chr(95)  # dummy, not used
        tags = []
        tags.append("system")
        lines = []
        sys_msg = "Ты — полезный ассистент. Отвечай на русском языке кратко и по существу."
        if self.filters:
            from groupcot.prompts import _lang_directive_from_filters
            lang_dir = _lang_directive_from_filters(self.filters, "output")
            if not lang_dir:
                lang_dir = _lang_directive_from_filters(self.filters, "input")
            if lang_dir:
                sys_msg += "\n\n" + lang_dir
        lines.append("<|im_start|>system")
        lines.append(sys_msg)
        lines.append("<|im_end|>")
        for msg in self._chat_history[-10:]:
            role = msg["role"]
            lines.append(f"<|im_start|>{role}")
            lines.append(msg["content"])
            lines.append("<|im_end|>")
        lines.append("<|im_start|>assistant")
        return "\n".join(lines)

    def _append_chat(self, role, text):
        self.chat_display.config(state="normal")
        prefix = "User: " if role == "user" else "Assistant: "
        self.chat_display.insert("end", prefix + text + "\n\n")
        self.chat_display.see("end")
        self.chat_display.config(state="disabled")

    def _clear_chat(self):
        self._chat_history.clear()
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.config(state="disabled")


def main():
    import sys

    if len(sys.argv) > 1:
        app = GroupGUI(config_path=sys.argv[1])
    else:
        app = GroupGUI()
    app.mainloop()


class _FilterDialog(tk.Toplevel):
    """Диалог добавления правила фильтрации."""

    def __init__(self, parent, pipeline="input", mode="text"):
        super().__init__(parent)
        self.title("Добавить правило фильтрации")
        self.resizable(False, False)
        self.result = None
        self._parent = parent

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Пайплайн:").grid(row=0, column=0, sticky="w", pady=4)
        self.pipeline_var = tk.StringVar(value=pipeline)
        ttk.Combobox(frm, textvariable=self.pipeline_var,
                     values=("input", "output", "feedback"),
                     state="readonly", width=14).grid(row=0, column=1, sticky="w")

        ttk.Label(frm, text="Режим:").grid(row=1, column=0, sticky="w", pady=4)
        self.mode_var = tk.StringVar(value=mode)
        ttk.Combobox(frm, textvariable=self.mode_var,
                     values=("text", "logit"),
                     state="readonly", width=14).grid(row=1, column=1, sticky="w")

        ttk.Label(frm, text="Тип:").grid(row=2, column=0, sticky="w", pady=4)
        self.type_var = tk.StringVar(value="language")
        ttk.Combobox(frm, textvariable=self.type_var,
                     values=("language", "topic", "pattern", "tag", "length"),
                     state="readonly", width=14).grid(row=2, column=1, sticky="w")

        ttk.Label(frm, text="Действие:").grid(row=3, column=0, sticky="w", pady=4)
        self.action_var = tk.StringVar(value="exclude")
        ttk.Combobox(frm, textvariable=self.action_var,
                     values=("exclude", "allow"),
                     state="readonly", width=14).grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="Значение:").grid(row=4, column=0, sticky="w", pady=4)
        self.value_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.value_var, width=30).grid(row=4, column=1, sticky="w")

        ttk.Label(frm, text="Описание:").grid(row=5, column=0, sticky="w", pady=4)
        self.desc_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.desc_var, width=30).grid(row=5, column=1, sticky="w")

        self.enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="включено", variable=self.enabled_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(frm, text="Зависит от (pipeline):").grid(row=7, column=0, sticky="w", pady=4)
        self.depends_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.depends_var, width=30).grid(row=7, column=1, sticky="w")

        hint = "language: ru/en/zh | pattern: regex | length: макс.символов | topic/tag: слово"
        ttk.Label(frm, text=hint, foreground="#666").grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=2)
        ttk.Button(btns, text="OK", command=self._ok).pack(side="left")
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side="left", padx=6)

        self.transient(parent)
        self.grab_set()

    def _ok(self):
        value = self.value_var.get().strip()
        if not value:
            messagebox.showwarning("Пустое значение", "Введите значение для правила", parent=self)
            return
        deps_raw = self.depends_var.get().strip()
        deps = [d.strip() for d in deps_raw.split(",") if d.strip()] if deps_raw else []
        self.result = FilterRule(
            pipeline=self.pipeline_var.get(),
            mode=self.mode_var.get(),
            type=self.type_var.get(),
            action=self.action_var.get(),
            value=value,
            description=self.desc_var.get().strip(),
            enabled=self.enabled_var.get(),
            depends_on=deps,
        )
        self.destroy()


class _ConceptDialog(tk.Toplevel):
    """Диалог добавления концепта для семантического конструктора (вкладка «Конструктор»)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Добавить концепт")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Интент (естественный язык):").grid(
            row=0, column=0, sticky="w", pady=4, columnspan=2)
        self.intent_text = tk.Text(frm, height=3, width=48, wrap="word")
        self.intent_text.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 8))

        ttk.Label(frm, text="Режим:").grid(row=2, column=0, sticky="w", pady=4)
        self.mode_var = tk.StringVar(value="exclude")
        ttk.Combobox(frm, textvariable=self.mode_var, values=("exclude", "attract"),
                     state="readonly", width=14).grid(row=2, column=1, sticky="w")

        self.semantic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="семантическое расширение (V3b, медленнее — строит индекс словаря)",
                        variable=self.semantic_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

        hint = ("exclude: убрать смысл из вывода | attract: притянуть к смыслу\n"
                "Пример: forbid the word 'cat'  /  говори только про горы")
        ttk.Label(frm, text=hint, foreground="#666", justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2)
        ttk.Button(btns, text="OK", command=self._ok).pack(side="left")
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side="left", padx=6)

        self.transient(parent)
        self.grab_set()

    def _ok(self):
        intent = self.intent_text.get("1.0", "end").strip()
        if not intent:
            messagebox.showwarning("Пустой интент", "Введите текст интента", parent=self)
            return
        self.result = {
            "intent": intent,
            "mode": self.mode_var.get(),
            "semantic": self.semantic_var.get(),
            "spec": None,
            "ids": None,
        }
        self.destroy()


class _ConceptGuardDialog(tk.Toplevel):
    """Диалог добавления guard-концепта для Фазы 4 (вкладка «Конструктор»).

    В отличие от `_ConceptDialog`, прототипы вводятся вручную — self-query не
    даёт надёжных целых предложений-примеров, а guard'у нужны именно они
    (ARCHITECTURE.md §5.1.1)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Добавить guard-концепт (Фаза 4)")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Название концепта:").grid(
            row=0, column=0, sticky="w", pady=4, columnspan=2)
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=54).grid(
            row=1, column=0, columnspan=2, sticky="we", pady=(0, 8))

        ttk.Label(frm, text="Прототипы (по одному предложению на строку):").grid(
            row=2, column=0, sticky="w", pady=4, columnspan=2)
        self.proto_text = tk.Text(frm, height=6, width=54, wrap="word")
        self.proto_text.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 8))

        hint = ("Пример (концепт «поездка к морю»):\n"
                "поехать на море\nпоехать на океан\nпоехать на рыбалку\nхочу искупаться в море\n\n"
                "Несколько по-разному сформулированных прототипов работают заметно лучше одного — "
                "иначе схожесть по шаблону фразы перевешивает схожесть по смыслу "
                "(проверено эмпирически, см. ARCHITECTURE.md §5.1.1).")
        ttk.Label(frm, text=hint, foreground="#666", justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2)
        ttk.Button(btns, text="OK", command=self._ok).pack(side="left")
        ttk.Button(btns, text="Отмена", command=self.destroy).pack(side="left", padx=6)

        self.transient(parent)
        self.grab_set()

    def _ok(self):
        name = self.name_var.get().strip()
        protos = [line.strip() for line in self.proto_text.get("1.0", "end").splitlines()
                  if line.strip()]
        if not name or not protos:
            messagebox.showwarning(
                "Заполните поля", "Нужны и название, и хотя бы один прототип", parent=self)
            return
        from .engine.constructor import ConceptSpec
        self.result = ConceptSpec(concept=name, mode="exclude", prototypes=protos)
        self.destroy()


if __name__ == "__main__":
    main()
