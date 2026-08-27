import json
import os
import queue
import subprocess
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

import httpx

ROOT = Path(__file__).resolve().parents[2]


class GroupGUI(tk.Tk):
    def __init__(self, config_path=None):
        super().__init__()
        self.title("GroupCOT — гипертекст-контекст на группах")
        self.geometry("1220x780")
        self.minsize(980, 620)

        self.cfg = load_config(config_path or (ROOT / "config.yaml"))
        self._q = queue.Queue()
        self._server_proc = None
        self._text_counter = 0

        self.group = Cyclic(64)
        self.store = Store(group=self.group)
        self.puller = Puller(self.store, top_k=2, threshold=0.0)
        self.state = ContextState(self.group)
        self.prompts = PromptSet()
        self.filters: list[FilterRule] = []

        self._build_ui()
        self.engine = self._create_default_engine()
        self.after(400, self._poll)
        self.after(800, self._auto_detect_server)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb

        self._build_model_tab(nb)
        self._build_group_tab(nb)
        self._build_store_tab(nb)
        self._build_run_tab(nb)
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
        for b in ("mock", "server", "llamacpp"):
            state = "normal" if (b != "llamacpp" or _llamacpp_ok) else "disabled"
            lbl = b if _llamacpp_ok or b != "llamacpp" else f"{b} (не установлен)"
            ttk.Radiobutton(frame, text=lbl, variable=self.backend_var, value=b, state=state).pack(anchor="w", padx=12)

        ttk.Label(frame, text="Путь GGUF (llamacpp):").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        self.model_path_var = tk.StringVar(value=self.cfg["model"]["path"])
        ttk.Entry(row, textvariable=self.model_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self._browse_model).pack(side="left")

        ttk.Label(frame, text="base_url (server):").pack(anchor="w", pady=(8, 0))
        self.base_url_var = tk.StringVar(value=self.cfg["model"]["base_url"])
        ttk.Entry(frame, textvariable=self.base_url_var).pack(fill="x")

        btns = ttk.Frame(frame)
        btns.pack(anchor="w", pady=8)
        ttk.Button(btns, text="Подключить движок", command=self._connect_engine).pack(side="left")
        ttk.Button(btns, text="Запустить llama-server", command=self._start_server).pack(side="left", padx=6)

        self.model_status = ttk.Label(frame, text="engine: не подключён")
        self.model_status.pack(anchor="w")

        ttk.Label(
            frame,
            text=(
                "Примечание: Qwen3-VL требует свежий llama.cpp (b6890+), сервер поднимается с --mmproj.\n"
                "ServerEngine ходит в /completions (grammar) и /embeddings того же llama-server.\n"
            "Порт 8080 может быть занят httpd — по умолчанию используется 8090."
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
                    n_gpu_layers=self.cfg["model"].get("n_gpu_layers", 0),
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

    def _auto_detect_server(self):
        """Попытаться подключиться к уже запущенному llama-server (только если engine ещё mock)."""
        if not isinstance(self.engine, MockEngine):
            return
        url = self.base_url_var.get().strip() or "http://127.0.0.1:8090"
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                engine = create_engine("server", base_url=url)
                engine.embed("ping")
                self.engine = engine
                self.backend_var.set("server")
                self.model_status.config(text=f"engine: server подключён (авто, {url})")
                return
        except Exception:
            pass
        if isinstance(self.engine, MockEngine):
            self.model_status.config(text="engine: mock (llamacpp не подключён — фильтры не применяются!)")

    def _check_server_health(self):
        """Периодическая проверка — жив ли сервер."""
        from groupcot.engine.server import ServerEngine
        if isinstance(self.engine, ServerEngine):
            url = self.engine.base_url
            try:
                r = httpx.get(f"{url}/health", timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            self.model_status.config(text=f"engine: server отвалился ({url})")
            self.engine = create_engine("mock", embed_dim=getattr(self.group, "dim", 8))
            return False
        return True

    def _connect_engine(self):
        backend = self.backend_var.get()
        try:
            if backend == "server":
                engine = create_engine("server", base_url=self.base_url_var.get().strip() or "http://127.0.0.1:8080")
                engine.embed("ping")
            elif backend == "llamacpp":
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

    def _start_server(self):
        cuda_exe = ROOT / "tools" / "cuda" / "llama-server.exe"
        cpu_exe = ROOT / "tools" / "llama-server.exe"
        if cuda_exe.exists() and (ROOT / "tools" / "cuda" / "ggml-cuda.dll").exists():
            exe = cuda_exe
            ngl = self.cfg["model"].get("n_gpu_layers", 99)
        elif cpu_exe.exists():
            exe = cpu_exe
            ngl = 0
        else:
            messagebox.showwarning(
                "llama-server.exe не найден",
                "Скачайте llama.cpp release и положите в tools\\ или tools\\cuda\\",
            )
            return
        model = ROOT / self.model_path_var.get()
        if not model.exists():
            messagebox.showwarning("Модель не найдена", f"Файл не существует: {model}")
            return
        cmd = [str(exe), "-m", str(model), "--port", "8090", "--embedding", "--pooling", "mean"]
        if ngl:
            cmd += ["-ngl", str(ngl)]
        mmproj = ROOT / "models" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
        if mmproj.exists():
            cmd += ["--mmproj", str(mmproj)]
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        with open(logs / "llama-server.log", "wb") as lf:
            self._server_proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW
            )
        self.model_status.config(text="llama-server запускается (лог: logs/llama-server.log)")
        self.after(5000, self._auto_detect_server)

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
        except queue.Empty:
            pass
        self._refresh_state()
        self._poll_count = getattr(self, "_poll_count", 0) + 1
        if self._poll_count % 15 == 0:
            self._check_server_health()
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
            prompt = self._build_chat_prompt()
            from groupcot.engine.server import ServerEngine
            from groupcot.engine.llamacpp import LlamaCppEngine

            logit_bias = None
            logits_processor = None
            blocked_ranges = None

            if isinstance(self.engine, ServerEngine):
                logit_bias = self._make_logit_bias()
            elif isinstance(self.engine, LlamaCppEngine):
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

            answer = self.engine.generate(prompt, max_tokens=512, temperature=0.7,
                                          logit_bias=logit_bias,
                                          logits_processor=logits_processor,
                                          blocked_ranges=blocked_ranges)

            badge = ""
            if filtered_langs:
                badge = f" [фильтр: {', '.join(filtered_langs)}]"

            self._q.put(("chat_reply", answer + badge))
        except Exception as exc:
            self._q.put(("chat_error", exc))

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

    def _make_logit_bias(self) -> dict[int, float] | None:
        """Построить logit_bias для ServerEngine из output-фильтров."""
        from groupcot.engine.server import ServerEngine
        if not isinstance(self.engine, ServerEngine):
            return None
        logit_rules = [f for f in self.filters if f.enabled and f.mode == "logit" and f.pipeline == "output"]
        if not logit_rules:
            return None
        vocab_size = 152064
        logit_bias: dict[int, float] = {}
        for rule in logit_rules:
            if rule.type == "language":
                try:
                    lang_ids = self.engine.build_lang_token_ids(rule.value, vocab_size)
                except Exception:
                    from groupcot.groups.token_group import TokenGroup
                    tg = TokenGroup(k=64)
                    lang_ids = tg.token_ids_for_lang(rule.value, vocab_size)
                bias = -100.0 if rule.action == "exclude" else 0.0
                for tid in lang_ids:
                    if tid < vocab_size:
                        logit_bias[tid] = bias
        return logit_bias if logit_bias else None

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


if __name__ == "__main__":
    main()
