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
  - Utilization for whichever GPUs you pick (Select GPUs...)

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
import ctypes
import tempfile
import subprocess
import threading
import urllib.request
import urllib.error
import tkinter as tk
from ctypes import wintypes
from tkinter import ttk
from collections import deque
from datetime import datetime

LOG_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Ollama\server.log")
OLLAMA_API = "http://127.0.0.1:11434/api/ps"
GPU_POLL_SECONDS = 2
MODEL_POLL_SECONDS = 2

# Which GPUs to display is a user choice that outlives a run, so it is kept
# outside the repo (the path is machine-specific and must never be committed).
CONFIG_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "OllamaMonitor", "config.json"
)

# --- Shared state for GPU usage + currently loaded models (separate lock, updated by their own threads) ---
aux_lock = threading.Lock()
aux_state = {
    "gpu_usage": {},           # {luid: max_utilization_pct}
    "gpu_error": None,
    "models": [],              # [{"name": "...", "vram_gb": 6.2}, ...]
    "models_error": None,
    "adapters": [],            # every adapter DXGI reports, annotated by the sampler
    "selected_gpus": [],       # the subset currently being displayed
    "selected_gpu_keys": None, # hardware keys the user picked; None = not loaded yet
    "selection_notice": None,  # set when the saved selection no longer fits the machine
}

# Set when the window is closed. Every worker loop watches it so they can unwind
# normally: Python kills daemon threads at interpreter exit WITHOUT running their
# finally blocks, which is how a sample file and a helper process got abandoned on
# every close.
shutdown = threading.Event()

# Guards the handle on the running typeperf child, so shutdown can reach in and
# stop it from whichever thread notices first.
sampler_lock = threading.Lock()
active_sampler = None

# Sample files this process has created and not yet deleted. The per-sample finally
# block normally clears each one, but it cannot run if a worker is still killed
# abruptly, so the exit path deletes whatever is left here by name - it cannot rely
# on the age-based sweep, which deliberately spares files this new.
our_samples = set()

GPU_SAMPLE_PREFIX = "gpu_sample_"
STALE_SAMPLE_AGE_SECONDS = 300


def terminate_active_sampler():
    """
    Stop the typeperf helper if one is mid-run.

    It is a separate process, so closing our window does not end it. Left alone
    with our end of its pipes gone it can block indefinitely rather than exiting
    on its own - one was found still resident 21 hours after the app had closed.
    """
    with sampler_lock:
        proc = active_sampler
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def final_cleanup():
    """
    Last pass on the way out: stop any surviving helper and delete the sample files
    this run created, by name. Called after the workers have been given their chance
    to unwind, so it normally finds nothing left to do.
    """
    terminate_active_sampler()
    with sampler_lock:
        leftovers = list(our_samples)
        our_samples.clear()
    for path in leftovers:
        try:
            os.remove(path)
        except OSError:
            pass
    purge_stale_gpu_samples()


def purge_stale_gpu_samples():
    """
    Delete sample files stranded by earlier runs.

    The routine cleanup is a finally block in sample_gpu_engine_counters; sweeping
    here as well is what actually clears the backlog, since that block is exactly
    what gets skipped when the app is killed mid-sample. Files younger than a few
    minutes are left alone in case a second copy of the monitor is running.
    """
    temp_dir = tempfile.gettempdir()
    cutoff = time.time() - STALE_SAMPLE_AGE_SECONDS
    try:
        names = os.listdir(temp_dir)
    except OSError:
        return

    for name in names:
        if not (name.startswith(GPU_SAMPLE_PREFIX) and name.endswith(".csv")):
            continue
        path = os.path.join(temp_dir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass  # held by another instance, or already gone - either is fine


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
        if shutdown.wait(1):
            return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)  # start at end of file, only show new lines
        while not shutdown.is_set():
            line = f.readline()
            if not line:
                # wait() rather than sleep() so a close is noticed immediately
                # instead of after the full poll interval.
                if shutdown.wait(0.1):
                    return
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
    while not shutdown.is_set():
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
        shutdown.wait(MODEL_POLL_SECONDS)


# --- DXGI adapter enumeration -------------------------------------------------
#
# Windows' performance counters identify a GPU only by LUID, and a LUID is handed
# out by the kernel per enumeration - the same card gets a different one after a
# reboot, a driver restart or a TDR recovery, and can even hold two at once (a
# stale entry lingers with no live counters behind it). Numbering GPUs by their
# position in a sorted LUID list is therefore not stable: anything that adds an
# adapter mid-session - connecting over RDP activates the software rasterizer,
# which then shows up in the counters - silently shifts every label down one.
#
# DXGI is the only interface that hands us something durable. We use it to turn
# each LUID into a real adapter name plus a vendor/device/subsystem triple that
# survives reboots, and key the saved GPU selection on that triple instead.
# LUIDs are treated strictly as this-boot-only handles for joining counter data.

DXGI_ADAPTER_FLAG_SOFTWARE = 2

# COM vtable slots. IUnknown occupies 0-2 and IDXGIObject 3-6, which puts
# IDXGIFactory1::EnumAdapters1 at 12 and IDXGIAdapter1::GetDesc1 at 10.
_VT_RELEASE = 2
_VT_ENUM_ADAPTERS1 = 12
_VT_GET_DESC1 = 10


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


_IID_IDXGIFactory1 = _GUID(
    0x770AAE78, 0xF26F, 0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


def _com_call(iface, slot, restype, argtypes, *args):
    """Invoke a COM method by vtable slot - ctypes has no COM support of its own."""
    vtable = ctypes.cast(iface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[slot])
    return fn(iface, *args)


def adapter_key(vendor_id, device_id, subsys_id):
    """Identity that survives reboots - what a saved selection is stored under."""
    return f"{vendor_id:04X}:{device_id:04X}:{subsys_id:08X}"


def short_name(description):
    """Trim vendor boilerplate so a name fits the compact readout row."""
    name = re.sub(r"\(R\)|\(TM\)|Corporation", "", description)
    name = re.sub(r"^\s*(NVIDIA|AMD|Intel|Microsoft)\s+", "", name.strip())
    return re.sub(r"\s{2,}", " ", name).strip() or description.strip()


def enumerate_adapters():
    """
    Every display adapter DXGI knows about, in enumeration order, as plain dicts.

    Returns [] rather than raising if DXGI is unavailable: parsing the log is the
    point of this tool and stays useful without GPU numbers, so a failure here
    must never be fatal.
    """
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []

    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) != 0:
        return []

    adapters = []
    try:
        index = 0
        while True:
            iface = ctypes.c_void_p()
            # A non-zero result is DXGI_ERROR_NOT_FOUND, marking the end of the list.
            hr = _com_call(factory, _VT_ENUM_ADAPTERS1, ctypes.c_long,
                           [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                           index, ctypes.byref(iface))
            if hr != 0:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                if _com_call(iface, _VT_GET_DESC1, ctypes.c_long,
                             [ctypes.POINTER(_DXGI_ADAPTER_DESC1)], ctypes.byref(desc)) == 0:
                    adapters.append({
                        # Spelled as two hex halves to match the counter instance
                        # names exactly, so the two sources can be joined on it.
                        "luid": f"{desc.AdapterLuid.HighPart:08X}_{desc.AdapterLuid.LowPart:08X}",
                        "name": short_name(desc.Description),
                        "key": adapter_key(desc.VendorId, desc.DeviceId, desc.SubSysId),
                        "subsys": f"{desc.SubSysId:08X}",
                        "vram_gb": desc.DedicatedVideoMemory / (1024 ** 3),
                        "is_software": bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                        "has_counters": False,  # filled in by the sampler each cycle
                    })
            finally:
                _com_call(iface, _VT_RELEASE, ctypes.c_ulong, [])
            index += 1
    finally:
        _com_call(factory, _VT_RELEASE, ctypes.c_ulong, [])
    return adapters


def real_adapters(adapters):
    """
    The adapters worth offering: physical hardware Windows is currently reporting
    counters for. Drops the software rasterizer, and drops stale duplicate entries
    (the same card under an older LUID) which enumerate fine but never emit data.
    """
    return [a for a in adapters if not a["is_software"] and a["has_counters"]]


def display_label(adapter, among):
    """Disambiguate identical cards by subsystem ID, but only when there are two."""
    if sum(1 for a in among if a["name"] == adapter["name"]) > 1:
        return f"{adapter['name']} ({adapter['subsys'][-4:]})"
    return adapter["name"]


def load_config():
    """Saved hardware keys, or None if there is no usable config yet."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            keys = json.load(f).get("selected_gpu_keys")
        return keys if isinstance(keys, list) else None
    except (OSError, ValueError):
        return None


def save_config(keys):
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"selected_gpu_keys": sorted(keys)}, f, indent=2)
    except OSError:
        pass  # a monitor that cannot save a preference is still a working monitor


def resolve_selection(adapters, saved_keys):
    """
    Turn saved hardware keys back into the adapters present right now, returning
    (selected, notice).

    A key that matches nothing - a swapped or removed card - falls back to every
    real adapter and says so in `notice`, rather than silently showing an empty
    row. An explicitly emptied selection is honoured as-is: that is a choice, not
    a mismatch.
    """
    candidates = real_adapters(adapters)
    if saved_keys is None:
        return candidates, None       # first run, nothing to reconcile
    if not saved_keys:
        return [], None               # user deliberately unchecked everything

    matched = [a for a in candidates if a["key"] in saved_keys]
    if not matched:
        return candidates, "Saved GPU selection not found on this system - showing all detected GPUs."

    missing = set(saved_keys) - {a["key"] for a in matched}
    if missing:
        return matched, f"{len(missing)} saved GPU(s) not detected - showing the rest."
    return matched, None


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

    Its stdout is discarded rather than piped. We never read it - the results come
    from the file - and an unread pipe is what let an orphaned typeperf wedge: once
    this process is gone nobody drains the buffer, so the child blocks on a write
    that will never complete instead of finishing its two samples and exiting.

    "Utilization Percentage" is a rate counter - it needs two samples to compute a
    percentage, so a single sample always comes back blank (a single space character,
    not zero). We request 2 samples one second apart and use the second, discarding
    the first.
    """
    global active_sampler

    tmp_path = None
    if shutdown.is_set():
        return {}  # closing: do not start another helper we would have to chase

    try:
        # Build a path without creating the file first - if it already exists, typeperf
        # silently prompts "overwrite? Y/N" on stdin and hangs forever waiting for input
        # we never send, since nothing is connected to answer it.
        tmp_path = os.path.join(tempfile.gettempdir(),
                                f"{GPU_SAMPLE_PREFIX}{uuid.uuid4().hex}.csv")

        with sampler_lock:
            our_samples.add(tmp_path)

        cmd = ["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-sc", "2", "-f", "CSV", "-o", tmp_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, creationflags=0x08000000)
        with sampler_lock:
            active_sampler = proc
            # A close can land in the gap between the loop's shutdown check and the
            # child being registered above - in which case terminate_active_sampler()
            # looked while this slot was still empty and found nothing to kill. Re-read
            # the flag here, inside the same lock, so the child cannot be missed.
            if shutdown.is_set():
                proc.kill()
        try:
            _, errors = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError("typeperf timed out")
        finally:
            with sampler_lock:
                active_sampler = None

        if proc.returncode != 0:
            raise RuntimeError((errors or "").strip() or "typeperf failed")

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
            # Upper-cased so it joins cleanly against DXGI, which always
            # formats its LUIDs in upper-case hex.
            luid = (m.group(1) + "_" + m.group(2)).upper()
            try:
                pct = float(val)
            except ValueError:
                continue
            per_luid_max[luid] = max(per_luid_max.get(luid, 0.0), pct)
        return per_luid_max
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # never created, or already gone
            with sampler_lock:
                our_samples.discard(tmp_path)


def gpu_usage_thread():
    """
    Sample GPU utilization, publish it keyed by LUID, and keep both the adapter list
    and the user's chosen subset in step with whatever the machine currently has.

    Adapters are re-enumerated every cycle rather than once at startup, because the
    set genuinely changes underneath us: connecting over RDP brings the software
    rasterizer to life, and a driver restart re-issues LUIDs. Re-reading is cheap
    and means the display corrects itself without needing a restart.

    During active generation, GPU engine instances get created/destroyed rapidly, which can
    cause a single typeperf sample to land mid-change and come back empty (not a real error).
    We retry briefly on an empty read, and otherwise keep showing the last known-good values
    rather than blanking the display for one bad cycle.
    """
    with aux_lock:
        aux_state["selected_gpu_keys"] = load_config()

    adapters = []
    while not shutdown.is_set():
        per_luid = None
        last_error = None
        for attempt in range(3):
            try:
                sample = sample_gpu_engine_counters()
                if sample:
                    per_luid = sample
                    last_error = None
                    break
                # empty-but-no-exception read: likely mid-change, retry quickly
                last_error = None
            except Exception as e:
                last_error = str(e)
            if shutdown.wait(0.3):
                return

        fresh = enumerate_adapters()
        if fresh:
            # On a cycle where sampling failed outright, carry the previous liveness
            # forward instead of declaring every adapter dead over one bad read.
            live = set(per_luid) if per_luid is not None else {
                a["luid"] for a in adapters if a["has_counters"]
            }
            for adapter in fresh:
                adapter["has_counters"] = adapter["luid"] in live
            adapters = fresh

        with aux_lock:
            selected, notice = resolve_selection(adapters, aux_state["selected_gpu_keys"])
            aux_state["adapters"] = adapters
            aux_state["selected_gpus"] = selected
            aux_state["selection_notice"] = notice
            if per_luid:
                aux_state["gpu_usage"] = per_luid
                aux_state["gpu_error"] = None
            elif last_error:
                # a real error occurred every attempt this cycle - surface it, but keep
                # last-known-good numbers on screen rather than wiping them
                aux_state["gpu_error"] = last_error
            # if the sample was empty and there was no error, just leave the previous
            # aux_state["gpu_usage"] value in place (transient glitch, not worth flagging)

        shutdown.wait(GPU_POLL_SECONDS)


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
        # The picker dialog sits on BG rather than BG_PANEL, so it needs its own variant.
        style.configure("Picker.TCheckbutton", background=self.BG, foreground=self.FG,
                         focuscolor=self.BG)
        style.map("Picker.TCheckbutton",
                  background=[("active", self.BG)],
                  indicatorcolor=[("selected", self.ACCENT), ("!selected", self.BG_PANEL)])

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

        # The GPU readouts are built at runtime rather than declared here: how many
        # there are, and which, depends on what the user picked and what is plugged
        # in right now. Two fixed labels is what made a hot-plugged adapter able to
        # push a real GPU off the display entirely.
        self.gpu_row = tk.Frame(top, bg=self.BG_PANEL)
        self.gpu_row.pack(side="top", fill="x")
        self._gpu_row_sig = None
        self._gpu_value_vars = {}

        self.notice_var = tk.StringVar(value="")
        self._label(top, textvariable=self.notice_var, size=8, muted=True).pack(
            side="top", anchor="w", padx=10)

        toggle_row = tk.Frame(top, bg=self.BG_PANEL)
        toggle_row.pack(side="top", fill="x")
        tk.Button(toggle_row, text="Select GPUs...", command=self._open_gpu_picker,
                  bg=self.BG_PANEL, fg=self.FG, activebackground=self.TREE_HEADER_BG,
                  activeforeground=self.FG, relief="flat", borderwidth=1,
                  font=("Segoe UI", 9), cursor="hand2").pack(side="left", padx=10, pady=(0, 8))
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
        # Closing the window has to stand the workers down explicitly - see _on_close.
        self._after_id = None
        root.protocol("WM_DELETE_WINDOW", self._on_close)

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

    def _on_close(self):
        """
        Stand the background work down before tearing the window down.

        Without this the interpreter exits the moment the window goes, killing the
        worker threads where they stand - so the typeperf child is orphaned and its
        temp file is never deleted. Signalling first gives both a chance to end
        cleanly; main() then waits briefly for the threads to finish unwinding.
        """
        shutdown.set()
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass  # already fired or the widget is gone; nothing to cancel
            self._after_id = None
        terminate_active_sampler()
        self.root.destroy()

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

    def _rebuild_gpu_row(self, selected):
        """Rebuild the per-GPU readouts. Only called when the set of displayed GPUs
        actually changes - tearing widgets down five times a second would flicker."""
        for child in self.gpu_row.winfo_children():
            child.destroy()
        self._gpu_value_vars = {}

        if not selected:
            self._label(self.gpu_row, "No GPUs selected", muted=True).pack(
                side="left", padx=10, pady=(0, 6))
            return

        for adapter in selected:
            cell = tk.Frame(self.gpu_row, bg=self.BG_PANEL)
            cell.pack(side="left", padx=(10, 8), pady=(0, 6))
            self._label(cell, f"{display_label(adapter, selected)}:", bold=True).pack(side="left")
            var = tk.StringVar(value="--")
            self._label(cell, textvariable=var).pack(side="left", padx=(5, 0))
            self._gpu_value_vars[adapter["luid"]] = var

    def _open_gpu_picker(self):
        """
        Let the user choose which adapters to display.

        The choice is stored by hardware key (vendor/device/subsystem), never by
        position in a list, so it survives reboots, driver restarts and an RDP
        session adding a software adapter partway down the enumeration.
        """
        with aux_lock:
            adapters = list(aux_state.get("adapters") or [])
            saved = aux_state.get("selected_gpu_keys")

        chosen = set(saved) if saved is not None else {a["key"] for a in real_adapters(adapters)}

        win = tk.Toplevel(self.root)
        win.title("Select GPUs")
        win.configure(bg=self.BG)
        win.transient(self.root)
        win.resizable(False, False)

        body = tk.Frame(win, bg=self.BG)
        body.pack(fill="both", expand=True, padx=16, pady=(14, 6))
        self._label(body, "Show utilization for:", bold=True, size=11).pack(anchor="w", pady=(0, 8))

        list_frame = tk.Frame(body, bg=self.BG)
        list_frame.pack(fill="both", expand=True)

        entries = []                        # [(adapter, BooleanVar)] currently on screen
        show_all = tk.BooleanVar(value=False)

        def capture():
            """Fold the visible checkboxes back into `chosen` before redrawing, so
            toggling 'show all' doesn't discard ticks made in the other view."""
            for adapter, var in entries:
                if var.get():
                    chosen.add(adapter["key"])
                else:
                    chosen.discard(adapter["key"])

        def render():
            for child in list_frame.winfo_children():
                child.destroy()
            entries.clear()

            visible = adapters if show_all.get() else real_adapters(adapters)
            if not visible:
                self._label(list_frame, "No GPUs detected yet - try again in a moment.",
                            muted=True).pack(anchor="w")
                return

            for adapter in visible:
                notes = []
                if adapter["is_software"]:
                    notes.append("software renderer")
                if not adapter["has_counters"]:
                    notes.append("no activity data")
                suffix = f"   [{', '.join(notes)}]" if notes else ""
                # Integrated GPUs report a fraction of a GB; rounding those to
                # "0 GB" reads as broken, so keep a decimal below 1.
                gb = adapter['vram_gb']
                vram = f"{gb:.0f} GB" if gb >= 1 else f"{gb:.1f} GB"
                label = f"{adapter['name']}  -  {vram}  -  {adapter['subsys']}{suffix}"
                var = tk.BooleanVar(value=adapter["key"] in chosen)
                entries.append((adapter, var))
                ttk.Checkbutton(list_frame, text=label, variable=var,
                                style="Picker.TCheckbutton").pack(anchor="w", pady=2)

        def on_toggle_all():
            capture()
            render()

        render()

        ttk.Checkbutton(body, text="Show all adapters (including software and inactive)",
                        variable=show_all, command=on_toggle_all,
                        style="Picker.TCheckbutton").pack(anchor="w", pady=(12, 0))

        def button(parent, text, command, accent=False):
            return tk.Button(parent, text=text, command=command,
                             bg=self.ACCENT if accent else self.BG_PANEL,
                             fg="#000000" if accent else self.FG,
                             activebackground=self.ACCENT if accent else self.TREE_HEADER_BG,
                             activeforeground="#000000" if accent else self.FG,
                             relief="flat", borderwidth=1, width=10,
                             font=("Segoe UI", 9), cursor="hand2")

        def save_and_close():
            capture()
            keys = sorted(chosen)
            save_config(keys)
            with aux_lock:
                aux_state["selected_gpu_keys"] = keys
            win.destroy()

        buttons = tk.Frame(win, bg=self.BG)
        buttons.pack(fill="x", padx=16, pady=(4, 14))
        button(buttons, "Save", save_and_close, accent=True).pack(side="right")
        button(buttons, "Cancel", win.destroy).pack(side="right", padx=(0, 8))

        # Centre on the parent so it can't open off-screen on a multi-monitor desk.
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.grab_set()

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
        selected = aux.get("selected_gpus") or []
        signature = tuple(a["luid"] for a in selected)
        if signature != self._gpu_row_sig:
            self._gpu_row_sig = signature
            self._rebuild_gpu_row(selected)

        gpu_usage = aux.get("gpu_usage") or {}
        for luid, var in self._gpu_value_vars.items():
            if luid in gpu_usage:
                var.set(f"{gpu_usage[luid]:.0f}%")
            elif aux.get("gpu_error"):
                var.set("n/a")
            else:
                var.set("--")

        self.notice_var.set(aux.get("selection_notice") or "")

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

        self._after_id = self.root.after(200, self.refresh)


def main():
    purge_stale_gpu_samples()  # clear anything a previous run left behind

    # Still daemons, so a wedged worker can never stop the app exiting - but they
    # are joined below so that in the normal case they unwind properly first.
    workers = [
        threading.Thread(target=parser_thread, daemon=True),
        threading.Thread(target=ollama_models_thread, daemon=True),
        threading.Thread(target=gpu_usage_thread, daemon=True),
    ]
    for worker in workers:
        worker.start()

    root = tk.Tk()
    app = MonitorGUI(root)
    try:
        root.mainloop()
    finally:
        shutdown.set()
        terminate_active_sampler()
        for worker in workers:
            # Long enough for a killed sampler to unwind its own finally block;
            # final_cleanup() covers whatever misses that window.
            worker.join(timeout=3)
        final_cleanup()


if __name__ == "__main__":
    main()
