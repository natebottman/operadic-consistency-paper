"""Tiny thread-safe RPM logger.

Each API request writes a single timestamp line to RPM_LOG. `current_rpm()`
counts timestamps in the trailing 60-second window. `wait_for_headroom()`
sleeps if the projected RPM would exceed `cap` after the next request.
"""
import os, time, threading

RPM_LOG = os.environ.get("RPM_LOG", "data/raw/rpm.log")
RPM_CAP = int(os.environ.get("RPM_CAP", "3000"))

_lock = threading.Lock()


def _ensure():
    d = os.path.dirname(RPM_LOG)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(RPM_LOG):
        open(RPM_LOG, "w").close()


def log_request():
    """Append the current timestamp to RPM_LOG (thread-safe)."""
    _ensure()
    with _lock:
        with open(RPM_LOG, "a") as f:
            f.write(f"{time.time()}\n")


def _read_timestamps():
    _ensure()
    try:
        with open(RPM_LOG) as f:
            out = []
            for line in f:
                try:
                    out.append(float(line.strip()))
                except ValueError:
                    pass
            return out
    except FileNotFoundError:
        return []


def current_rpm(window: float = 60.0) -> int:
    """Number of requests in the last `window` seconds."""
    cutoff = time.time() - window
    return sum(1 for t in _read_timestamps() if t >= cutoff)


def peak_rpm(window: float = 60.0, step: float = 5.0) -> int:
    """Peak RPM (sliding `window`s computed every `step` seconds)."""
    ts = sorted(_read_timestamps())
    if not ts:
        return 0
    t0, t1 = ts[0], ts[-1]
    peak = 0
    t = t0
    while t <= t1:
        cnt = sum(1 for x in ts if t - window < x <= t)
        if cnt > peak:
            peak = cnt
        t += step
    return peak


def wait_for_headroom(cap: int = None, headroom: int = 50, poll: float = 0.5):
    """Block until current_rpm < cap-headroom. Cheap pre-call throttle."""
    if cap is None:
        cap = RPM_CAP
    while current_rpm() >= cap - headroom:
        time.sleep(poll)
