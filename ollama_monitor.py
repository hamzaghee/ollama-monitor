"""
Ollama Live Performance Monitor
--------------------------------
Tails the Ollama server.log in real time, parses llama.cpp's slot timing
lines, and displays live stats in a small always-on-top window:
  - Status (idle / processing prompt / generating)
  - Context used (n_tokens / n_ctx_slot)
  - Prompt processing speed (tokens/sec)
  - Time to first token (TTFT) for the current request
  - Live generation speed (instant + 3s rolling average)
  - Tokens generated so far in the current request

Requires only Python's standard library (tkinter ships with the standard
Windows Python installer) - no extra pip installs needed.

Run with:  python ollama_monitor.py
"""

import os
import re
import csv
import io
import json
import time
import uuid
import tempfile
import subprocess
import threading
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk
from collections import deque
from datetime import datetime

LOG_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Ollama\server.log")
OLLAMA_API = "http://127.0.0.1:11434/api/ps"
GPU_POLL_SECONDS = 2
MODEL_POLL_SECONDS = 2

# --- Shared state for GPU usage + currently loaded models (separate lock, updated by their own threads) ---
aux_lock = threading.Lock()
aux_state = {
    "gpu_usage": {},       # {"GPU 0": 12.3, "GPU 1": 0.0}
    "gpu_error": None,
    "models": [],          # [{"name": "...", "vram_gb": 6.2}, ...]
    "models_error": None,
}

# --- Regex patterns matching llama.cpp's slot timing log lines ---
RE_NEW_PROMPT = re.compile(
    r"new prompt, n_ctx_slot = (?P<n_ctx>\d+), n_keep = \d+, task\.n_tokens = (?P<n_tokens>\d+)"
)
RE_PROMPT_PROGRESS = re.compile(
    r"prompt processing, n_tokens\s*=\s*(?P<n_tokens>\d+), progress = (?P<progress>[\d.]+), "
    r"t\s*=\s*(?P<t>[\d.]+) s / (?P<tps>[\d.]+) tokens per second"
)
RE_GEN = re.compile(
    r"n_gen\s*=\s*(?P<n_gen>\d+), tg\s*=\s*(?P<tg>[\d.]+) t/s, tg_3s\s*=\s*(?P<tg3s>[\d.]+) t/s"
)

# --- Shared state, guarded by a lock (writer thread + GUI thread) ---
state_lock = threading.Lock()
state = {
    "status": "Waiting for activity...",
    "n_ctx": None,
    "n_tokens": None,
    "prompt_tps": None,
    "prompt_progress": None,
    "n_gen": None,
    "tg": None,
    "tg3s": None,
    "ttft": None,
    "last_update": None,
    "request_start": None,
    "first_gen_seen": False,
}

# --- History of completed requests (most recent first) ---
HISTORY_MAXLEN = 25
history = deque(maxlen=HISTORY_MAXLEN)


def snapshot_to_history(s, model_name):
    """Turn a finished request's state snapshot into a history row, if it has data worth logging."""
    if s.get("n_gen") is None:
        return  # this "request" never actually generated anything, skip it
    history.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "model": model_name,
        "ttft": s.get("ttft"),
        "tg3s": s.get("tg3s"),
        "n_gen": s.get("n_gen"),
        "n_tokens": s.get("n_tokens"),
        "n_ctx": s.get("n_ctx"),
    })


def tail_log(path):
    """Generator that yields new lines appended to `path`, like `tail -f`."""
    while not os.path.exists(path):
        time.sleep(1)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)  # start at end of file, only show new lines
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            yield line.rstrip("\n")


def parser_thread():
    for line in tail_log(LOG_PATH):
        now = time.time()

        m = RE_NEW_PROMPT.search(line)
        if m:
            with aux_lock:
                current_models = aux_state.get("models") or []
            model_name = current_models[0]["name"] if current_models else "unknown"
            with state_lock:
                snapshot_to_history(state, model_name)
                state["status"] = "Processing prompt..."
                state["n_ctx"] = int(m.group("n_ctx"))
                state["n_tokens"] = int(m.group("n_tokens"))
                state["prompt_tps"] = None
                state["prompt_progress"] = 0.0
                state["n_gen"] = None
                state["tg"] = None
                state["tg3s"] = None
                state["ttft"] = None
                state["request_start"] = now
                state["first_gen_seen"] = False
                state["last_update"] = now
            continue

        m = RE_PROMPT_PROGRESS.search(line)
        if m:
            with state_lock:
                state["status"] = "Processing prompt..."
                state["n_tokens"] = int(m.group("n_tokens"))
                state["prompt_progress"] = float(m.group("progress"))
                state["prompt_tps"] = float(m.group("tps"))
                state["last_update"] = now
            continue

        m = RE_GEN.search(line)
        if m:
            with state_lock:
                if not state["first_gen_seen"] and state["request_start"]:
                    state["ttft"] = now - state["request_start"]
                    state["first_gen_seen"] = True
                state["status"] = "Generating..."
                state["n_gen"] = int(m.group("n_gen"))
                state["tg"] = float(m.group("tg"))
                state["tg3s"] = float(m.group("tg3s"))
                state["last_update"] = now
            continue


def ollama_models_thread():
    """Poll Ollama's own /api/ps endpoint to see which model(s) are currently loaded and their VRAM footprint."""
    while True:
        try:
            with urllib.request.urlopen(OLLAMA_API, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("models", []):
                vram_bytes = m.get("size_vram", 0) or 0
                models.append({
                    "name": m.get("name") or m.get("model") or "unknown",
                    "vram_gb": vram_bytes / (1024 ** 3),
                })
            with aux_lock:
                aux_state["models"] = models
                aux_state["models_error"] = None
        except Exception as e:
            with aux_lock:
                aux_state["models"] = []
                aux_state["models_error"] = str(e)
        time.sleep(MODEL_POLL_SECONDS)


# Matches the LUID pair inside a GPU Engine / GPU Process Memory counter instance name, e.g.
# "...luid_0x00000000_0x0000ABCD_phys_0_eng_0_engtype_3D)"
LUID_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)")


def sample_gpu_engine_counters():
    """
    Run Windows' built-in "GPU Engine" performance counter (the same data source Task
    Manager's Performance tab reads from) and return {luid: max_utilization_pct}.
    Windows reports one utilization value per engine (3D, Compute, Copy, etc.) per process;
    Task Manager's single "GPU 0 / GPU 1" percentage is the highest of those per adapter, so
    we do the same here rather than summing (summing would double count parallel engines).

    typeperf writes to a temp file rather than being captured from stdout: when run without
    a real console window (which we do intentionally, to avoid flashing a black window), it
    can assume a narrow display width and insert line breaks mid-row, corrupting the CSV
    structure. Writing straight to a file avoids that entirely.

    "Utilization Percentage" is a rate counter - it needs two samples to compute a
    percentage, so a single sample always comes back blank (a single space character,
    not zero). We request 2 samples one second apart and use the second, discarding
    the first.
    """
    tmp_path = None
    try:
        # Build a path without creating the file first - if it already exists, typeperf
        # silently prompts "overwrite? Y/N" on stdin and hangs forever waiting for input
        # we never send, since nothing is connected to answer it.
        tmp_path = os.path.join(tempfile.gettempdir(), f"gpu_sample_{uuid.uuid4().hex}.csv")

        cmd = ["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-sc", "2", "-f", "CSV", "-o", tmp_path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, creationflags=0x08000000)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "typeperf failed")

        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        reader = csv.reader(io.StringIO(content))
        rows = [r for r in reader if r]  # drop any blank rows
        if len(rows) < 3:  # header + 2 data rows expected
            raise RuntimeError("no counter data returned")

        headers, values = rows[0], rows[-1]  # last row = second (valid) sample
        per_luid_max = {}
        for header, val in zip(headers, values):
            m = LUID_RE.search(header)
            if not m:
                continue
            luid = m.group(1) + "_" + m.group(2)
            try:
                pct = float(val)
            except ValueError:
                continue
            per_luid_max[luid] = max(per_luid_max.get(luid, 0.0), pct)
        return per_luid_max
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def gpu_usage_thread():
    """
    Periodically sample GPU utilization and publish it as GPU 0 / GPU 1 (ordered by LUID so
    the labels stay consistent run-to-run; this ordering isn't guaranteed to match Device
    Manager's numbering, but will typically stay stable on the same machine).

    During active generation, GPU engine instances get created/destroyed rapidly, which can
    cause a single typeperf sample to land mid-change and come back empty (not a real error).
    We retry briefly on an empty read, and otherwise keep showing the last known-good values
    rather than blanking the display for one bad cycle.
    """
    while True:
        labeled = {}
        last_error = None
        for attempt in range(3):
            try:
                per_luid = sample_gpu_engine_counters()
                if per_luid:
                    ordered_luids = sorted(per_luid.keys())
                    labeled = {f"GPU {i}": per_luid[luid] for i, luid in enumerate(ordered_luids)}
                    last_error = None
                    break
                # empty-but-no-exception read: likely mid-change, retry quickly
                last_error = None
            except Exception as e:
                last_error = str(e)
            time.sleep(0.3)

        with aux_lock:
            if labeled:
                aux_state["gpu_usage"] = labeled
                aux_state["gpu_error"] = None
            elif last_error:
                # a real error occurred every attempt this cycle - surface it, but keep
                # last-known-good numbers on screen rather than wiping them
                aux_state["gpu_error"] = last_error
            # if labeled is empty and there was no error, just leave the previous
            # aux_state["gpu_usage"] value in place (transient glitch, not worth flagging)

        time.sleep(GPU_POLL_SECONDS)


def fmt(val, suffix="", digits=1):
    if val is None:
        return "--"
    if isinstance(val, float):
        return f"{val:.{digits}f}{suffix}"
    return f"{val}{suffix}"


class MonitorGUI:
    # --- Dark theme palette ---
    BG = "#1e1e1e"
    BG_PANEL = "#252526"
    FG = "#e0e0e0"
    FG_MUTED = "#8a8a8a"
    ACCENT = "#4fc1ff"
    BORDER = "#3a3d41"
    TREE_HEADER_BG = "#2d2d2d"
    TREE_SELECT_BG = "#094771"

    def __init__(self, root):
        self.root = root
        root.title("Ollama Live Monitor")
        root.geometry("760x460")
        root.minsize(520, 380)
        root.resizable(True, True)
        root.configure(bg=self.BG)

        style = ttk.Style()
        style.theme_use("clam")  # required for full color control over ttk widgets on Windows
        style.configure("TProgressbar", troughcolor=self.BG_PANEL, background=self.ACCENT,
                         bordercolor=self.BG, lightcolor=self.ACCENT, darkcolor=self.ACCENT)
        style.configure("Treeview", background=self.BG_PANEL, foreground=self.FG,
                         fieldbackground=self.BG_PANEL, bordercolor=self.BORDER, borderwidth=0)
        style.configure("Treeview.Heading", background=self.TREE_HEADER_BG, foreground=self.FG,
                         relief="flat", borderwidth=1)
        style.map("Treeview", background=[("selected", self.TREE_SELECT_BG)],
                  foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading", background=[("active", self.TREE_HEADER_BG)])
        style.configure("TCheckbutton", background=self.BG_PANEL, foreground=self.FG,
                         focuscolor=self.BG_PANEL)
        style.map("TCheckbutton",
                  background=[("active", self.BG_PANEL)],
                  indicatorcolor=[("selected", self.ACCENT), ("!selected", self.BG)])

        # --- Top strip: loaded model(s) + live GPU utilization (row 1), toggle (row 2) ---
        # The toggle gets its own row so it's never pushed off-screen by the info row
        # when the window gets narrow - two short rows always fit better than one long one.
        top = tk.Frame(root, bg=self.BG_PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        top.pack(side="top", fill="x", padx=10, pady=(10, 0))

        info_row = tk.Frame(top, bg=self.BG_PANEL)
        info_row.pack(side="top", fill="x")
        info_row.grid_columnconfigure(1, weight=1)  # model name gets whatever space is left

        self._label(info_row, "Model:", bold=True).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
        self.model_var = tk.StringVar(value="--")
        self.model_label = self._label(info_row, textvariable=self.model_var)
        self.model_label.grid(row=0, column=1, sticky="w", pady=6)

        self._label(info_row, "GPU 0:", bold=True).grid(row=0, column=2, sticky="w", padx=(20, 4), pady=6)
        self.gpu0_var = tk.StringVar(value="--")
        self._label(info_row, textvariable=self.gpu0_var).grid(row=0, column=3, sticky="w", pady=6)

        self._label(info_row, "GPU 1:", bold=True).grid(row=0, column=4, sticky="w", padx=(20, 4), pady=6)
        self.gpu1_var = tk.StringVar(value="--")
        self._label(info_row, textvariable=self.gpu1_var).grid(row=0, column=5, sticky="w", padx=(0, 8), pady=6)

        toggle_row = tk.Frame(top, bg=self.BG_PANEL)
        toggle_row.pack(side="top", fill="x")
        self.topmost_var = tk.BooleanVar(value=False)
        topmost_toggle = ttk.Checkbutton(toggle_row, text="Always on top", variable=self.topmost_var,
                                          command=self._toggle_topmost, style="TCheckbutton")
        topmost_toggle.pack(side="right", padx=10, pady=(0, 8))

        # --- Middle area: left = live stats, right = history ---
        middle = tk.Frame(root, bg=self.BG)
        middle.pack(side="top", fill="both", expand=True)

        # left's width is set explicitly (not left to its children's natural size) so it
        # can actually shrink when the window does - see _apply_responsive_layout below.
        self.left_frame = tk.Frame(middle, bg=self.BG, width=300)
        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(False)
        left = self.left_frame

        self.left_metric_labels = []

        def add_metric_label(textvariable, size=10, bold=False, muted=False):
            lbl = self._label(left, textvariable=textvariable, size=size, bold=bold, muted=muted)
            lbl.pack(padx=12, pady=6, anchor="w", fill="x")
            self.left_metric_labels.append(lbl)
            return lbl

        self.status_var = tk.StringVar(value="Waiting for activity...")
        add_metric_label(self.status_var, size=13, bold=True)

        self.ctx_var = tk.StringVar(value="Context: --")
        add_metric_label(self.ctx_var)
        self.ctx_bar = ttk.Progressbar(left, length=300, maximum=100)
        self.ctx_bar.pack(fill="x", padx=12, pady=(0, 10))

        self.prompt_var = tk.StringVar(value="Prompt eval: --")
        add_metric_label(self.prompt_var)

        self.ttft_var = tk.StringVar(value="Time to first token: --")
        add_metric_label(self.ttft_var)

        self.tg_var = tk.StringVar(value="Generation speed: --")
        add_metric_label(self.tg_var, size=12, bold=True)

        self.ngen_var = tk.StringVar(value="Tokens generated: --")
        add_metric_label(self.ngen_var)

        self.idle_var = tk.StringVar(value="")
        add_metric_label(self.idle_var, size=8, muted=True)

        # --- Right panel: history of past requests ---
        right = tk.Frame(middle, bg=self.BG)
        right.pack(side="left", fill="both", expand=True, padx=(8, 12), pady=12)

        self._label(right, "Previous runs", bold=True, size=11).pack(anchor="w")

        columns = ("time", "model", "ttft", "tps", "tokens", "context")
        self.history_tree = ttk.Treeview(right, columns=columns, show="headings", height=14)
        self.history_tree.heading("time", text="Time")
        self.history_tree.heading("model", text="Model")
        self.history_tree.heading("ttft", text="TTFT")
        self.history_tree.heading("tps", text="Tok/s")
        self.history_tree.heading("tokens", text="Tokens")
        self.history_tree.heading("context", text="Context")
        self.history_tree.column("time", width=70, anchor="center")
        self.history_tree.column("model", width=140, anchor="center")
        self.history_tree.column("ttft", width=60, anchor="center")
        self.history_tree.column("tps", width=60, anchor="center")
        self.history_tree.column("tokens", width=65, anchor="center")
        self.history_tree.column("context", width=90, anchor="center")
        self.history_tree.pack(fill="both", expand=True)

        self._last_history_len = -1

        # Recalculate wrap widths whenever the window itself is resized, so long text
        # (the model name, long metric lines) wraps instead of getting clipped, and the
        # left panel actually shrinks instead of refusing to go below its natural size.
        root.bind("<Configure>", self._on_root_configure)
        root.update_idletasks()
        self._apply_responsive_layout(root.winfo_width())

        self.refresh()

    def _label(self, parent, text=None, textvariable=None, size=10, bold=False, muted=False):
        """Small helper so every label picks up the dark theme colors consistently.
        anchor/justify are set on the label itself (not just its pack position) since
        those control where text sits *within* the label's own box - without them the
        text still renders centered even when the label is anchored left in its parent."""
        kwargs = {
            "bg": parent.cget("bg"),
            "fg": self.FG_MUTED if muted else self.FG,
            "font": ("Segoe UI", size, "bold" if bold else "normal"),
            "anchor": "w",
            "justify": "left",
        }
        if textvariable is not None:
            kwargs["textvariable"] = textvariable
        else:
            kwargs["text"] = text
        return tk.Label(parent, **kwargs)

    def _toggle_topmost(self):
        self.root.attributes("-topmost", self.topmost_var.get())

    def _on_root_configure(self, event):
        if event.widget is self.root:
            self._apply_responsive_layout(event.width)

    def _apply_responsive_layout(self, window_width):
        """Keep the model name and left-panel metrics readable as the window is resized:
        wrap long text instead of clipping it, and let the left panel's width track the
        window instead of staying pinned to its widest unwrapped line."""
        model_wrap = max(140, window_width - 340)
        self.model_label.configure(wraplength=model_wrap)

        left_width = max(220, min(420, int(window_width * 0.42)))
        self.left_frame.configure(width=left_width)
        label_wrap = max(140, left_width - 24)
        for lbl in self.left_metric_labels:
            lbl.configure(wraplength=label_wrap)

    def refresh(self):
        with state_lock:
            s = dict(state)
        with aux_lock:
            aux = dict(aux_state)

        # Model(s) currently loaded in Ollama
        models = aux.get("models") or []
        if aux.get("models_error"):
            self.model_var.set("unavailable")
        elif models:
            self.model_var.set(", ".join(f"{m['name']} ({m['vram_gb']:.1f} GB)" for m in models))
        else:
            self.model_var.set("none loaded")

        # Live GPU utilization - prefer showing the last known-good reading over blanking
        # the display on a transient error (see gpu_usage_thread for why these happen).
        gpu_usage = aux.get("gpu_usage") or {}
        if not gpu_usage and aux.get("gpu_error"):
            self.gpu0_var.set("n/a")
            self.gpu1_var.set("n/a")
        else:
            self.gpu0_var.set(f"{gpu_usage.get('GPU 0', 0):.0f}%" if "GPU 0" in gpu_usage else "--")
            self.gpu1_var.set(f"{gpu_usage.get('GPU 1', 0):.0f}%" if "GPU 1" in gpu_usage else "--")

        self.status_var.set(s["status"])

        if s["n_tokens"] is not None and s["n_ctx"]:
            pct = min(100, (s["n_tokens"] / s["n_ctx"]) * 100)
            self.ctx_bar["value"] = pct
            self.ctx_var.set(f"Context: {s['n_tokens']} / {s['n_ctx']} tokens ({pct:.0f}%)")
        else:
            self.ctx_bar["value"] = 0
            self.ctx_var.set("Context: --")

        if s["prompt_tps"] is not None:
            prog_pct = (s["prompt_progress"] or 0) * 100
            self.prompt_var.set(f"Prompt eval: {fmt(s['prompt_tps'])} tok/s ({prog_pct:.0f}% done)")
        else:
            self.prompt_var.set("Prompt eval: --")

        self.ttft_var.set(f"Time to first token: {fmt(s['ttft'], ' s', 2)}")

        if s["tg"] is not None:
            self.tg_var.set(f"Generation speed: {fmt(s['tg'])} tok/s  (3s avg: {fmt(s['tg3s'])})")
        else:
            self.tg_var.set("Generation speed: --")

        self.ngen_var.set(f"Tokens generated: {fmt(s['n_gen'])}")

        if s["last_update"]:
            idle_for = time.time() - s["last_update"]
            if idle_for > 5:
                self.idle_var.set(f"No new activity for {idle_for:.0f}s")
            else:
                self.idle_var.set("")

        # Update history table only when it's actually changed (avoids flicker/rebuild every tick)
        if len(history) != self._last_history_len:
            self._last_history_len = len(history)
            self.history_tree.delete(*self.history_tree.get_children())
            for row in history:
                ctx_str = f"{row['n_tokens']}/{row['n_ctx']}" if row["n_tokens"] and row["n_ctx"] else "--"
                self.history_tree.insert("", "end", values=(
                    row["time"],
                    row.get("model", "unknown"),
                    fmt(row["ttft"], "s", 2),
                    fmt(row["tg3s"]),
                    fmt(row["n_gen"]),
                    ctx_str,
                ))

        self.root.after(200, self.refresh)


def main():
    threading.Thread(target=parser_thread, daemon=True).start()
    threading.Thread(target=ollama_models_thread, daemon=True).start()
    threading.Thread(target=gpu_usage_thread, daemon=True).start()

    root = tk.Tk()
    app = MonitorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
