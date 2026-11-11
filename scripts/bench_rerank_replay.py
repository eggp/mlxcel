#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay recorded rerank requests against one server binary and compare runs.

Starts the given mlxcel-server binary, sends every request of a local JSON
Lines file to POST /v1/rerank, and writes one JSON result file with, per
request, the client wall time, the peak and the retained physical footprint of
the server process, the HTTP status and the scores in document order, and for
the whole run the GPU time of the server process. A second run against another
binary or configuration gives a second file, and `compare` reports how the two
differ, ranking changes included.

The request file is private. Its text is sent to the local server and nowhere
else: result files and console output carry ids, counts, timings, memory
figures and scores only. The server's own stdout and stderr are discarded for
the same reason.

Each input line is one JSON object:

  {"id": "optional-string", "query": "...", "documents": ["...", "..."]}

Without `id` the 1-based line number identifies the request.

Memory is the physical footprint (`ri_phys_footprint` from `proc_pid_rusage`),
read through ctypes several times per second. RSS is not used: it does not
include the Metal GPU buffers. Off macOS the footprint fields are null and the
run records `"footprint_source": "unavailable"`.

GPU time is the sum of `accumulatedGPUTime` (ns) over the server process's
Metal clients in `ioreg -r -w 0 -c AGXDeviceUserClient`, read once the server
is ready and again after the last request; the run records the difference.
When `ioreg` is missing, fails, or lists no client of the server process, the
value is null and `gpu_time_source` says why.

Modes:
  replay      one request at a time
  concurrent  the same with N requests in flight
  startup     K fresh server starts, the first request sent twice per start
  abandon     how long a small request waits behind a large one whose caller
              hung up
  compare     two replay or concurrent result files, or two abandon ones

Examples:
  scripts/bench_rerank_replay.py replay --bin target/release/mlxcel-server \\
      --server-args="-m <reranker dir> --reranker-model <reranker dir>" \\
      --requests ~/private/rerank-requests.jsonl --commit 0ba1b50b \\
      --live-traffic no --out ~/private/replay-base.json

  scripts/bench_rerank_replay.py compare --before ~/private/replay-base.json \\
      --after ~/private/replay-fix.json

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import ctypes
import http.client
import json
import math
import os
import re
import select
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

HOST = "127.0.0.1"
READY_TIMEOUT_S = 900.0
READY_POLL_S = 0.2
REQUEST_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 60.0
# How long a server whose connection just broke is given to finish exiting, so
# that its death is seen at that request and not at a later one.
EXIT_SETTLE_S = 1.0
SAMPLE_INTERVAL_S = 0.1
# Environment overrides whose values are recorded. Any other name may hold a
# secret, so only the name is written.
RECORDED_ENV_PREFIXES = ("MLXCEL_", "MLX_")
TOP_CUTS = (5, 10)


class InputError(Exception):
    """A request file or a result file that cannot be used."""


class ServerError(Exception):
    """The server binary could not be run, did not become ready, or exited."""


# --- physical footprint ------------------------------------------------------

RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` from <sys/resource.h>."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


def _load_proc_pid_rusage():
    """Return libproc's `proc_pid_rusage`, or None where it does not exist."""
    if sys.platform != "darwin":
        return None
    try:
        function = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


_PROC_PID_RUSAGE = _load_proc_pid_rusage()
FOOTPRINT_SOURCE = "proc_pid_rusage" if _PROC_PID_RUSAGE else "unavailable"


def _rusage(pid: int) -> _RusageInfoV4 | None:
    if _PROC_PID_RUSAGE is None:
        return None
    info = _RusageInfoV4()
    if _PROC_PID_RUSAGE(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info


def read_footprint(pid: int) -> int | None:
    """Physical footprint of `pid` in bytes; None when it cannot be read."""
    info = _rusage(pid)
    return None if info is None else int(info.ri_phys_footprint)


def read_lifetime_max_footprint(pid: int) -> int | None:
    """Largest physical footprint `pid` ever had, in bytes; None when unknown."""
    info = _rusage(pid)
    return None if info is None else int(info.ri_lifetime_max_phys_footprint)


class FootprintSampler:
    """Samples one process's footprint and keeps the peak of every open window."""

    def __init__(self, pid: int):
        self._pid = pid
        self._lock = threading.Lock()
        self._open: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "FootprintSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_S):
            self.sample()

    def sample(self) -> int | None:
        """Read the footprint once and fold it into every open window."""
        value = read_footprint(self._pid)
        if value is not None:
            with self._lock:
                for window in self._open:
                    peak = window["peak"]
                    window["peak"] = value if peak is None else max(peak, value)
        return value

    @contextlib.contextmanager
    def window(self):
        """Yield a dict that holds `peak` and, once closed, `retained`.

        The footprint is read when the window opens and when it closes, so a
        request shorter than the sampling interval still has two readings.
        """
        window = {"peak": None, "retained": None}
        with self._lock:
            self._open.append(window)
        self.sample()
        try:
            yield window
        finally:
            window["retained"] = self.sample()
            with self._lock:
                self._open.remove(window)


# --- GPU time ----------------------------------------------------------------

# Every Metal client of a process is one AGXDeviceUserClient entry; its
# `AppUsage` list, which can be empty or absent, holds the GPU time it used.
IOREG_COMMAND = ("ioreg", "-r", "-w", "0", "-c", "AGXDeviceUserClient")
IOREG_TIMEOUT_S = 60.0
GPU_TIME_SOURCE = "ioreg"
_CLIENT_CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+),')
_ACCUMULATED_GPU_TIME = re.compile(r'"accumulatedGPUTime"=(\d+)')


def gpu_time_in(listing: str, pid: int) -> int | None:
    """Sum of `accumulatedGPUTime` over the clients of `pid` in an ioreg
    listing, in ns; None when the listing has no client of `pid`."""
    total = None
    for client in listing.split("+-o ")[1:]:
        creator = _CLIENT_CREATOR.search(client)
        if creator is None or int(creator.group(1)) != pid:
            continue
        used = sum(int(ns) for ns in _ACCUMULATED_GPU_TIME.findall(client))
        total = used if total is None else total + used
    return total


def read_gpu_time(pid: int) -> tuple[int | None, str]:
    """GPU time `pid` has used so far, in ns, and its source; the value is
    None and the source says why when it cannot be read."""
    try:
        done = subprocess.run(
            list(IOREG_COMMAND),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=IOREG_TIMEOUT_S,
        )
    except FileNotFoundError:
        return None, "unavailable: ioreg not found"
    except (OSError, subprocess.TimeoutExpired):
        return None, "unavailable: ioreg failed"
    if done.returncode != 0:
        return None, "unavailable: ioreg failed"
    total = gpu_time_in(done.stdout, pid)
    if total is None:
        return None, "unavailable: no GPU client of the server process"
    return total, GPU_TIME_SOURCE


def gpu_time_fields(start: tuple, end: tuple) -> dict:
    """The run's `gpu_time_ns`, `end` minus `start` (two `read_gpu_time`
    results), and `gpu_time_source`, the reason of the first failed read."""
    for value, source in (start, end):
        if value is None:
            return {"gpu_time_ns": None, "gpu_time_source": source}
    return {"gpu_time_ns": end[0] - start[0], "gpu_time_source": GPU_TIME_SOURCE}


# --- request file ------------------------------------------------------------


@dataclass(frozen=True)
class RerankRequest:
    id: str | int
    query: str
    documents: list[str]


def is_string_list(value) -> bool:
    """True for a non-empty list of strings."""
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


def rerank_request(item: dict, request_id: str | int, where: str) -> RerankRequest:
    if not isinstance(item.get("query"), str):
        raise InputError(f"{where}: `query` must be a string")
    if not is_string_list(item.get("documents")):
        raise InputError(f"{where}: `documents` must be a non-empty array of strings")
    return RerankRequest(request_id, item["query"], item["documents"])


def read_request_file(path: str, make) -> list:
    """Read and validate a request file, one JSON object per line.

    `make(item, request_id, where)` builds the request from one object or
    raises InputError. Error messages name the file and the line, never the
    line's content.
    """
    requests = []
    seen: set = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{line_no}"
                try:
                    item = json.loads(line)
                except ValueError:
                    raise InputError(f"{where}: not valid JSON") from None
                if not isinstance(item, dict):
                    raise InputError(f"{where}: not a JSON object")
                request_id = item.get("id", line_no)
                request = make(item, request_id, where)
                if "id" in item and not isinstance(request_id, str):
                    raise InputError(f"{where}: `id` must be a string")
                if request_id in seen:
                    raise InputError(f"{where}: duplicate request id")
                seen.add(request_id)
                requests.append(request)
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: cannot read: {type(exc).__name__}") from None
    if not requests:
        raise InputError(f"{path}: no requests")
    return requests


def load_requests(path: str) -> list[RerankRequest]:
    return read_request_file(path, rerank_request)


# --- server process ----------------------------------------------------------


def binary_version(binary: str) -> str:
    """Output of `<binary> --version`."""
    try:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServerError(f"cannot run {binary} --version: {exc}") from None
    return (done.stdout or done.stderr).strip()


def accepts_connections(port: int) -> bool:
    try:
        socket.create_connection((HOST, port), timeout=2).close()
    except OSError:
        return False
    return True


def answers_http(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=5)
        try:
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            response.read()
        finally:
            conn.close()
    except (OSError, http.client.HTTPException):
        return False
    return response.status == 200


def wait_ready(proc: subprocess.Popen, port: int, probe) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServerError(f"server exited early with code {proc.returncode}")
        if probe(port):
            return
        time.sleep(READY_POLL_S)
    raise ServerError(f"server did not become ready within {READY_TIMEOUT_S:g} s")


def require_running(proc: subprocess.Popen, settle: bool = False) -> None:
    """Raise ServerError once the started server has exited.

    A port answers for whichever process holds it, so after the started
    server is gone a response would come from another process, and the
    request text would go to it.
    """
    if settle:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=EXIT_SETTLE_S)
    if proc.poll() is not None:
        raise ServerError(
            f"server exited with code {proc.returncode} during the run; "
            "no result file is written"
        )


# Held for every liveness check, and from the check before a request until
# that request is sent. A worker therefore cannot find the server dead while
# another one stands between its own check and its send.
SEND_LOCK = threading.Lock()


def post_to_running(proc: subprocess.Popen, port: int, request, send=None, read=None):
    """Send one request to the started server, refused when it is gone.

    `send(port, request)` and `read(conn, request)` speak the endpoint; they
    default to `send_rerank` and `read_rerank`. Returns (wall_ms, status,
    result, error) and raises ServerError when the server is found dead
    before the send or after the response. The wall time starts once the
    lock is held, so waiting for another worker is not in it.
    """
    send, read = send or send_rerank, read or read_rerank
    with SEND_LOCK:
        require_running(proc)
        start = time.perf_counter()
        conn, error = send(port, request)
    status = result = None
    if conn is not None:
        status, result, error = read(conn, request)
    wall_ms = round((time.perf_counter() - start) * 1000.0, 3)
    with SEND_LOCK:
        require_running(proc, settle=status is None)
    return wall_ms, status, result, error


def abandon_request(
    proc: subprocess.Popen, port: int, request, delay_s: float, send=None, read=None
) -> float | None:
    """Send one request and hang up `delay_s` seconds later, reading nothing.

    `send` and `read` are those of `post_to_running`. Returns the delay
    applied, in ms from the request sent to the connection closed. Returns
    None when the server answered within the delay: that response is read
    and dropped, and the request was not abandoned. Raises ServerError when
    the server is found dead, or the request cannot be sent.
    """
    send, read = send or send_rerank, read or read_rerank
    with SEND_LOCK:
        require_running(proc)
        conn, error = send(port, request)
        sent = time.perf_counter()
    if conn is None:
        with SEND_LOCK:
            require_running(proc, settle=True)
        raise ServerError(
            f"the request to abandon could not be sent ({error}); "
            "no result file is written"
        )
    remaining = delay_s
    while remaining > 0:
        if select.select([conn.sock], [], [], remaining)[0]:
            status, _, _ = read(conn, request)
            with SEND_LOCK:
                require_running(proc, settle=status is None)
            return None
        remaining = sent + delay_s - time.perf_counter()
    conn.close()
    return round((time.perf_counter() - sent) * 1000.0, 3)


def require_free_port(port: int) -> None:
    """Refuse a port that already accepts connections.

    Readiness is probed through the port, and a listener that is already
    there would answer for the server.
    """
    if accepts_connections(port):
        raise ServerError(
            f"port {port} on {HOST} already accepts connections; "
            "the server needs a free port"
        )


@contextlib.contextmanager
def running_server(args: argparse.Namespace, probe):
    """Start the server, wait for `probe`, and stop that same process on exit.

    The port is checked again here: time has passed since the check before
    the run, and `startup` starts a server more than once.
    """
    require_free_port(args.port)
    command = [args.bin, *args.server_args, "--host", HOST, "--port", str(args.port)]
    try:
        proc = subprocess.Popen(
            command,
            env={**os.environ, **dict(args.env)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise ServerError(f"cannot start {args.bin}: {exc}") from None
    try:
        wait_ready(proc, args.port, probe)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# --- one request -------------------------------------------------------------


def is_finite_number(value) -> bool:
    """True for an int or a float that a strict JSON number can hold.

    A boolean is an int in Python and is not a number here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def scores_in_document_order(payload, document_count: int) -> list[float]:
    """Scores of a /v1/rerank response, indexed like the request's documents.

    The server returns results sorted by score; a comparison needs them by
    document. Raises ValueError unless every document has exactly one score.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != document_count:
        raise ValueError("results do not match the documents")
    scores: list[float | None] = [None] * document_count
    for result in results:
        index = result.get("index") if isinstance(result, dict) else None
        score = result.get("relevance_score") if isinstance(result, dict) else None
        if (
            type(index) is not int
            or not 0 <= index < document_count
            or scores[index] is not None
            or not is_finite_number(score)
        ):
            raise ValueError("malformed result entry")
        scores[index] = float(score)
    return scores


def send_json(port: int, path: str, payload: dict):
    """Connect and POST `payload` to `path`. Returns (connection, error).

    The connection is None when the request could not be sent. `error` is a
    short category, here and in `read_json`, and never response text, which
    may echo the request.
    """
    body = json.dumps(payload)
    conn = None
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=REQUEST_TIMEOUT_S)
        conn.request(
            "POST",
            path,
            body=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    except (OSError, http.client.HTTPException) as exc:
        if conn is not None:
            conn.close()
        return None, f"transport:{type(exc).__name__}"
    return conn, None


def read_json(conn: http.client.HTTPConnection, parse):
    """Read the response to a sent request and close the connection.

    A 200 body is decoded and handed to `parse`, which raises ValueError for
    an unusable one. Returns (status, parsed, error).
    """
    try:
        try:
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()
    except (OSError, http.client.HTTPException) as exc:
        return None, None, f"transport:{type(exc).__name__}"
    if response.status != 200:
        return response.status, None, "http_status"
    try:
        return 200, parse(json.loads(raw)), None
    except ValueError:
        return 200, None, "bad_response"


def send_rerank(port: int, request: RerankRequest):
    return send_json(
        port, "/v1/rerank", {"query": request.query, "documents": request.documents}
    )


def read_rerank(conn: http.client.HTTPConnection, request: RerankRequest):
    """Returns (status, scores, error)."""
    return read_json(
        conn, lambda payload: scores_in_document_order(payload, len(request.documents))
    )


def measure_request(
    proc: subprocess.Popen,
    port: int,
    request: RerankRequest,
    sampler: FootprintSampler,
    in_flight: int,
) -> dict:
    with sampler.window() as window:
        wall_ms, status, scores, error = post_to_running(proc, port, request)
    return {
        "id": request.id,
        "documents": len(request.documents),
        "in_flight": in_flight,
        "wall_ms": wall_ms,
        "footprint_peak_bytes": window["peak"],
        "footprint_retained_bytes": window["retained"],
        "status": status,
        "error": error,
        "scores": scores,
    }


# --- modes -------------------------------------------------------------------


def mib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value / 1024**2:.1f}"


def write_result(path: str, result: dict) -> None:
    text = json.dumps(result, indent=2, allow_nan=False)
    try:
        with open(path, "w", encoding="utf-8") as out:
            out.write(text + "\n")
    except OSError as exc:
        raise InputError(f"{path}: cannot write: {exc.strerror}") from None
    print(f"wrote {path}")


def require_writable(path: str) -> None:
    """Refuse an output path that cannot be written, changing nothing.

    An existing file is opened for writing without truncation, so it keeps
    its content until the run has succeeded. For a new file a temporary file
    is created beside it and removed again.
    """
    if not path:
        raise InputError("--out: the output path is empty")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise InputError(f"{path}: the directory does not exist")
    try:
        try:
            os.close(os.open(path, os.O_WRONLY))
        except FileNotFoundError:
            handle, probe = tempfile.mkstemp(dir=directory)
            try:
                os.close(handle)
            finally:
                os.unlink(probe)
    except OSError as exc:
        raise InputError(f"{path}: cannot write: {exc.strerror}") from None


def run_record(args: argparse.Namespace) -> dict:
    """What is recorded once per run. Before the binary is executed at all,
    even for its version, refuses an output path that cannot be written and a
    port that is in use."""
    require_writable(args.out)
    require_free_port(args.port)
    env = dict(args.env)
    return {
        "mode": args.mode,
        "date": time.strftime("%Y-%m-%d"),
        "binary": os.path.abspath(args.bin),
        "binary_version": binary_version(args.bin),
        "commit": args.commit,
        "live_traffic": args.live_traffic,
        "server_args": args.server_args,
        "env_names": sorted(env),
        "env_values": {
            name: value
            for name, value in sorted(env.items())
            if name.startswith(RECORDED_ENV_PREFIXES)
        },
        "footprint_source": FOOTPRINT_SOURCE,
    }


def run_replay(args: argparse.Namespace) -> int:
    in_flight = getattr(args, "in_flight", 1)
    requests = load_requests(args.requests)
    run = run_record(args)
    run["in_flight"] = in_flight
    with running_server(args, answers_http) as proc:
        gpu_start = read_gpu_time(proc.pid)
        with FootprintSampler(proc.pid) as sampler:

            def one(request: RerankRequest) -> dict:
                row = measure_request(proc, args.port, request, sampler, in_flight)
                print(
                    f"id={row['id']} docs={row['documents']} "
                    f"status={row['status']} wall_ms={row['wall_ms']:.1f} "
                    f"peak_mib={mib(row['footprint_peak_bytes'])} "
                    f"retained_mib={mib(row['footprint_retained_bytes'])}",
                    flush=True,
                )
                return row

            with ThreadPoolExecutor(max_workers=in_flight) as pool:
                rows = list(pool.map(one, requests))
        run.update(gpu_time_fields(gpu_start, read_gpu_time(proc.pid)))
        run["footprint_lifetime_max_bytes"] = read_lifetime_max_footprint(proc.pid)
    failed = sum(1 for row in rows if row["scores"] is None)
    print(f"{len(rows) - failed} of {len(rows)} requests scored, {failed} failed")
    write_result(args.out, {"run": run, "requests": rows})
    return 1 if failed else 0


def run_startup(args: argparse.Namespace) -> int:
    request = load_requests(args.requests)[0]
    run = run_record(args)
    run["page_cache"] = args.page_cache
    rows = []
    failed = 0
    for start in range(1, args.starts + 1):
        with running_server(args, accepts_connections) as proc:
            before = read_footprint(proc.pid)
            first_ms, first_status, first_scores, first_error = post_to_running(
                proc, args.port, request
            )
            after = read_footprint(proc.pid)
            second_ms, second_status, second_scores, second_error = post_to_running(
                proc, args.port, request
            )
        failed += first_scores is None or second_scores is None
        rows.append(
            {
                "start": start,
                "id": request.id,
                "documents": len(request.documents),
                "first_wall_ms": first_ms,
                "first_status": first_status,
                "first_error": first_error,
                "second_wall_ms": second_ms,
                "second_status": second_status,
                "second_error": second_error,
                "footprint_before_bytes": before,
                "footprint_after_first_bytes": after,
            }
        )
        print(
            f"start={start} first_ms={first_ms:.1f} status={first_status} "
            f"second_ms={second_ms:.1f} status={second_status} "
            f"before_mib={mib(before)} after_first_mib={mib(after)}",
            flush=True,
        )
    write_result(args.out, {"run": run, "starts": rows})
    return 1 if failed else 0


# The keys of a row `run_abandon` writes, after `id` and the count of what the
# request holds.
ABANDON_MEASURE_KEYS = (
    "abandoned_after_ms",
    "alone_wall_ms",
    "alone_status",
    "alone_error",
    "behind_wall_ms",
    "behind_status",
    "behind_error",
    "idle_status",
    "idle_error",
    "footprint_retained_bytes",
)
ABANDON_ROW_KEYS = ("id", "documents", *ABANDON_MEASURE_KEYS)
ABANDON_PROBES = ("alone", "behind", "idle")


def probe_failed(row: dict) -> bool:
    return any(row[f"{probe}_error"] is not None for probe in ABANDON_PROBES)


def run_abandon(args: argparse.Namespace) -> int:
    requests = load_requests(args.requests)
    if len(requests) < 2:
        raise InputError(
            f"{args.requests}: abandon needs at least two requests, "
            "a probe and a request to abandon"
        )
    large = [r for r in requests if len(r.documents) >= args.min_documents]
    if not large:
        raise InputError(
            f"{args.requests}: no request has at least {args.min_documents} documents"
        )
    # `min` keeps the first of several smallest requests.
    probe = min(requests, key=lambda request: len(request.documents))
    if len(probe.documents) >= args.min_documents:
        raise InputError(
            f"{args.requests}: the smallest request's document count is "
            f"{len(probe.documents)} and --min-documents is {args.min_documents}; "
            "the probe must be smaller than a large request"
        )
    run = run_record(args)
    run["in_flight"] = 1
    run["abandon_after_s"] = args.abandon_after
    run["min_documents"] = args.min_documents
    run["probe_id"] = probe.id
    run["probe_documents"] = len(probe.documents)
    run["large_requests"] = len(large)
    rows = []
    with running_server(args, answers_http) as proc:
        gpu_start = read_gpu_time(proc.pid)
        for request in large:
            alone_ms, alone_status, _, alone_error = post_to_running(proc, args.port, probe)
            abandoned_after_ms = abandon_request(proc, args.port, request, args.abandon_after)
            behind_ms, behind_status, _, behind_error = post_to_running(proc, args.port, probe)
            retained = read_footprint(proc.pid)
            # A server that kept scoring the abandoned request has finished it
            # once the probe behind it and one more probe have been answered,
            # if it serves rerank requests in arrival order on one worker.
            _, idle_status, _, idle_error = post_to_running(proc, args.port, probe)
            rows.append(
                {
                    "id": request.id,
                    "documents": len(request.documents),
                    "abandoned_after_ms": abandoned_after_ms,
                    "alone_wall_ms": alone_ms,
                    "alone_status": alone_status,
                    "alone_error": alone_error,
                    "behind_wall_ms": behind_ms,
                    "behind_status": behind_status,
                    "behind_error": behind_error,
                    "idle_status": idle_status,
                    "idle_error": idle_error,
                    "footprint_retained_bytes": retained,
                }
            )
            delay = (
                "abandoned=no"
                if abandoned_after_ms is None
                else f"abandoned_after_ms={abandoned_after_ms:.1f}"
            )
            print(
                f"id={request.id} docs={len(request.documents)} {delay} "
                f"alone_ms={alone_ms:.1f} status={alone_status} "
                f"behind_ms={behind_ms:.1f} status={behind_status} "
                f"idle_status={idle_status} retained_mib={mib(retained)}",
                flush=True,
            )
        run.update(gpu_time_fields(gpu_start, read_gpu_time(proc.pid)))
        run["footprint_lifetime_max_bytes"] = read_lifetime_max_footprint(proc.pid)
    abandoned = sum(1 for row in rows if row["abandoned_after_ms"] is not None)
    failed = sum(1 for row in rows if probe_failed(row))
    print(
        f"{len(rows)} large requests: {abandoned} abandoned, "
        f"{len(rows) - abandoned} not abandoned, {failed} with a failed probe"
    )
    write_result(args.out, {"run": run, "requests": rows})
    return 1 if failed else 0


# --- compare -----------------------------------------------------------------


# The keys of a row `measure_request` writes.
REQUEST_ROW_KEYS = (
    "id",
    "documents",
    "in_flight",
    "wall_ms",
    "footprint_peak_bytes",
    "footprint_retained_bytes",
    "status",
    "error",
    "scores",
)


def entry_defect(row, keys: tuple, numbers: tuple, count: str = "documents") -> str | None:
    """What is wrong with `row` in the part every mode's rows share, or None:
    every key in `keys` present, a usable `id`, an integer under `count`, and
    each key in `numbers` null or a finite number."""
    if not isinstance(row, dict):
        return "not a JSON object"
    for name in keys:
        if name not in row:
            return f"`{name}` is missing"
    request_id = row["id"]
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        return "`id` must be a string or an integer"
    if type(row[count]) is not int:
        return f"`{count}` must be an integer"
    for name in numbers:
        if row[name] is not None and not is_finite_number(row[name]):
            return f"`{name}` must be null or a finite number"
    return None


def abandon_row_defect(row, count: str = "documents") -> str | None:
    """What makes an `abandon` row unusable for a comparison, or None.

    The error fields decide whether a probe failed, so each must be null or a
    string, and a probe without a wall time must be a failed one: a compared
    row then has both wall times.
    """
    defect = entry_defect(
        row,
        ("id", count, *ABANDON_MEASURE_KEYS),
        ("abandoned_after_ms", "alone_wall_ms", "behind_wall_ms", "footprint_retained_bytes"),
        count,
    )
    if defect:
        return defect
    for probe in ABANDON_PROBES:
        error = row[f"{probe}_error"]
        if error is not None and not isinstance(error, str):
            return f"`{probe}_error` must be null or a string"
    for probe in ("alone", "behind"):
        if row[f"{probe}_wall_ms"] is None:
            if row[f"{probe}_status"] == 200:
                return f"`{probe}_wall_ms` is null although `{probe}_status` is 200"
            if row[f"{probe}_error"] is None:
                return f"`{probe}_wall_ms` is null although `{probe}_error` is null too"
    return None


def row_defect(row) -> str | None:
    """What makes `row` unusable for a comparison, or None.

    A row has every key in REQUEST_ROW_KEYS. It is a recorded failure
    (`scores` null) or carries one finite score per document. The replay path never scores a request without documents,
    so an empty score list is refused rather than compared.
    """
    defect = entry_defect(
        row, REQUEST_ROW_KEYS, ("wall_ms", "footprint_peak_bytes", "footprint_retained_bytes")
    )
    if defect:
        return defect
    documents, scores = row["documents"], row["scores"]
    if scores is None:
        return None
    if not isinstance(scores, list) or not scores:
        return "`scores` must be null or a non-empty array"
    if not all(is_finite_number(score) for score in scores):
        return "`scores` must hold finite numbers only"
    if documents != len(scores):
        return "`documents` does not match the number of scores"
    return None


# The check each comparable mode's rows must pass.
ROW_DEFECTS = {
    "replay": row_defect,
    "concurrent": row_defect,
    "abandon": abandon_row_defect,
}


def load_result(path: str, row_defects: dict = ROW_DEFECTS) -> dict:
    """Read and validate a replay, concurrent or abandon result file.

    `row_defects` holds the row check of each mode. A file written before
    GPU time was recorded has no `gpu_time_ns`; where present it is null or
    an integer. Error messages name the file and the entry's position, never
    its content.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"{path}: cannot read: {type(exc).__name__}") from None
    except ValueError:
        raise InputError(f"{path}: not valid JSON") from None
    run = data.get("run") if isinstance(data, dict) else None
    rows = data.get("requests") if isinstance(data, dict) else None
    mode = run.get("mode") if isinstance(run, dict) else None
    if not isinstance(mode, str) or mode not in row_defects or not isinstance(rows, list):
        raise InputError(f"{path}: not a replay, concurrent or abandon result file")
    gpu_time = run.get("gpu_time_ns")
    if gpu_time is not None and type(gpu_time) is not int:
        raise InputError(f"{path}: `run.gpu_time_ns` must be null or an integer")
    if not isinstance(run.get("gpu_time_source", ""), str):
        raise InputError(f"{path}: `run.gpu_time_source` must be a string")
    for position, row in enumerate(rows, start=1):
        defect = row_defects[mode](row)
        if defect:
            raise InputError(f"{path}: request entry {position}: {defect}")
    counts = collections.Counter(row["id"] for row in rows)
    repeated = sum(1 for count in counts.values() if count > 1)
    if repeated:
        raise InputError(
            f"{path}: {repeated} ids appear more than once among {len(rows)} request entries"
        )
    return data


def spread(values: list[float]) -> dict | None:
    """min, p5, median, p95 and max of `values`; None when there are none."""
    if not values:
        return None
    if len(values) == 1:
        p5 = p95 = values[0]
    else:
        cuts = statistics.quantiles(values, n=20, method="inclusive")
        p5, p95 = cuts[0], cuts[-1]
    return {
        "n": len(values),
        "min": min(values),
        "p5": p5,
        "median": statistics.median(values),
        "p95": p95,
        "max": max(values),
    }


def top_set(scores: list[float], cut: int) -> set[int]:
    """Indices of the `cut` best documents, ordered as the server orders them:
    descending score, ties by ascending index."""
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return set(ranked[:cut])


def pair_rows(before: dict, after: dict, result: str, mismatch: str) -> tuple[dict, list]:
    """Pair the rows of two replay or concurrent result files by id.

    Returns (report, pairs). The report counts the ids present in one file
    only, those whose `result` is null on a side, and, under `mismatch`, those
    whose `result` lists differ in length; none of them is paired. For the
    (before, after) rows in `pairs` it holds the spreads of the wall time and
    footprint differences, after minus before.
    """
    before_rows = {row["id"]: row for row in before["requests"]}
    after_rows = {row["id"]: row for row in after["requests"]}
    shared = [request_id for request_id in before_rows if request_id in after_rows]
    deltas: dict[str, list[float]] = {
        "wall_ms": [],
        "footprint_peak_bytes": [],
        "footprint_retained_bytes": [],
    }
    failed = mismatched = 0
    pairs = []
    for request_id in shared:
        b, a = before_rows[request_id], after_rows[request_id]
        if not isinstance(b.get(result), list) or not isinstance(a.get(result), list):
            failed += 1
            continue
        if len(b[result]) != len(a[result]):
            mismatched += 1
            continue
        pairs.append((b, a))
        for name, values in deltas.items():
            if b.get(name) is not None and a.get(name) is not None:
                values.append(a[name] - b[name])
    report = {
        "only_before": len(before_rows) - len(shared),
        "only_after": len(after_rows) - len(shared),
        "failed_on_a_side": failed,
        mismatch: mismatched,
        "compared": len(pairs),
    }
    for name, values in deltas.items():
        report[name] = spread(values)
    return report, pairs


def compare_gpu_time(before: dict, after: dict) -> dict:
    """The run GPU time of each file and after minus before, in ns; null
    where a file has none."""
    b, a = before["run"].get("gpu_time_ns"), after["run"].get("gpu_time_ns")
    return {
        "gpu_time_before_ns": b,
        "gpu_time_after_ns": a,
        "gpu_time_diff_ns": None if b is None or a is None else a - b,
    }


def compare_runs(before: dict, after: dict) -> dict:
    """Differences, after minus before, over the requests both runs scored."""
    report, pairs = pair_rows(before, after, "scores", "document_count_mismatch")
    max_abs_score_diff = None
    changed = {cut: 0 for cut in TOP_CUTS}
    for b, a in pairs:
        diff = max(abs(x - y) for x, y in zip(a["scores"], b["scores"]))
        max_abs_score_diff = diff if max_abs_score_diff is None else max(max_abs_score_diff, diff)
        for cut in TOP_CUTS:
            if top_set(b["scores"], cut) != top_set(a["scores"], cut):
                changed[cut] += 1
    compared = report["compared"]
    report["max_abs_score_diff"] = max_abs_score_diff
    for cut in TOP_CUTS:
        report[f"top{cut}_changed_share"] = changed[cut] / compared if compared else None
    report.update(compare_gpu_time(before, after))
    return report


def compare_abandon_runs(before: dict, after: dict) -> dict:
    """The probe's wall time behind an abandoned request and alone, per file
    and as after minus before, over the large requests that were abandoned in
    both runs with every probe answered."""
    before_rows = {row["id"]: row for row in before["requests"]}
    after_rows = {row["id"]: row for row in after["requests"]}
    shared = [request_id for request_id in before_rows if request_id in after_rows]
    values: dict[str, list[float]] = {
        f"{probe}_{side}": []
        for probe in ("behind", "alone")
        for side in ("before", "after", "diff")
    }
    failed = not_abandoned = compared = 0
    for request_id in shared:
        b, a = before_rows[request_id], after_rows[request_id]
        if probe_failed(b) or probe_failed(a):
            failed += 1
            continue
        if b["abandoned_after_ms"] is None or a["abandoned_after_ms"] is None:
            not_abandoned += 1
            continue
        compared += 1
        for probe in ("behind", "alone"):
            b_ms, a_ms = b[f"{probe}_wall_ms"], a[f"{probe}_wall_ms"]
            values[f"{probe}_before"].append(b_ms)
            values[f"{probe}_after"].append(a_ms)
            values[f"{probe}_diff"].append(a_ms - b_ms)
    report = {
        "only_before": len(before_rows) - len(shared),
        "only_after": len(after_rows) - len(shared),
        "failed_on_a_side": failed,
        "not_abandoned_on_a_side": not_abandoned,
        "compared": compared,
    }
    for name, series in values.items():
        report[name] = spread(series)
    report.update(compare_gpu_time(before, after))
    return report


def format_spread(
    label: str, stats: dict | None, scale: float, unit: str, sign: str = "+"
) -> str:
    """One line for a distribution; `sign=""` for values that are not differences."""
    if stats is None:
        return f"{label}: no data"
    parts = "  ".join(
        f"{key} {stats[key] / scale:{sign}.2f}" for key in ("min", "p5", "median", "p95", "max")
    )
    return f"{label} ({unit}, n={stats['n']}): {parts}"


def print_headers(args: argparse.Namespace, before: dict, after: dict, names: tuple) -> None:
    """One line per file: its path, the `run` fields in `names`, the row count."""
    for label, path, data in (("before", args.before, before), ("after", args.after, after)):
        run = data["run"]
        fields = " ".join(f"{name}={run.get(name)}" for name in names)
        print(f"{label}: {path}  {fields} requests={len(data['requests'])}")


def print_deltas(report: dict) -> None:
    """The wall time and footprint spreads of a `pair_rows` report."""
    print()
    print("differences are after minus before")
    print(format_spread("wall time", report["wall_ms"], 1.0, "ms"))
    print(format_spread("peak footprint", report["footprint_peak_bytes"], 1024**2, "MiB"))
    print(
        format_spread(
            "retained footprint", report["footprint_retained_bytes"], 1024**2, "MiB"
        )
    )


def print_gpu_time(report: dict, before: dict, after: dict) -> None:
    """The `compare_gpu_time` part of a report, with each file's source."""
    print()
    print("server GPU time over the whole run")
    for side, data in (("before", before), ("after", after)):
        value = report[f"gpu_time_{side}_ns"]
        source = data["run"].get("gpu_time_source", "not recorded")
        shown = "n/a" if value is None else f"{value / 1e6:.2f} ms"
        print(f"{side}: {shown} ({source})")
    diff = report["gpu_time_diff_ns"]
    print("after minus before: " + ("n/a" if diff is None else f"{diff / 1e6:+.2f} ms"))


def print_abandon_comparison(
    args: argparse.Namespace, before: dict, after: dict, count: str = "documents"
) -> None:
    print_headers(
        args,
        before,
        after,
        ("commit", "date", "abandon_after_s", f"min_{count}", f"probe_{count}", "live_traffic"),
    )
    report = compare_abandon_runs(before, after)
    print()
    print(
        f"compared: {report['compared']} large requests abandoned in both files, "
        "every probe answered"
    )
    print(
        f"only in one file: {report['only_before']} before, {report['only_after']} after"
    )
    print(f"not abandoned in at least one file: {report['not_abandoned_on_a_side']}")
    print(f"failed probe in at least one file: {report['failed_on_a_side']}")
    for probe, title in (
        ("behind", "probe wall time behind an abandoned request"),
        ("alone", "probe wall time alone, the noise reference"),
    ):
        print()
        print(title)
        print(format_spread("before", report[f"{probe}_before"], 1.0, "ms", sign=""))
        print(format_spread("after", report[f"{probe}_after"], 1.0, "ms", sign=""))
        print(format_spread("after minus before", report[f"{probe}_diff"], 1.0, "ms"))
    print_gpu_time(report, before, after)


# The `run` fields `compare` prints for a replay or concurrent file.
REPLAY_HEADER_FIELDS = ("commit", "date", "in_flight", "live_traffic")


def load_pair(args: argparse.Namespace, row_defects: dict = ROW_DEFECTS) -> tuple[dict, dict, bool]:
    """Load `--before` and `--after`. Returns (before, after, abandon), where
    `abandon` says both are abandon results; refuses a pair in which only one
    of them is."""
    before, after = load_result(args.before, row_defects), load_result(args.after, row_defects)
    modes = [data["run"]["mode"] for data in (before, after)]
    if "abandon" in modes and modes != ["abandon", "abandon"]:
        raise InputError(
            f"{args.before} holds {modes[0]} results and {args.after} holds "
            f"{modes[1]} results; abandon results compare only with other "
            "abandon results"
        )
    return before, after, "abandon" in modes


def run_compare(args: argparse.Namespace) -> int:
    before, after, abandon = load_pair(args)
    if abandon:
        print_abandon_comparison(args, before, after)
        return 0
    print_headers(args, before, after, REPLAY_HEADER_FIELDS)
    report = compare_runs(before, after)
    print()
    print(f"compared: {report['compared']} requests scored in both files")
    print(
        f"only in one file: {report['only_before']} before, {report['only_after']} after"
    )
    print(f"failed in at least one file: {report['failed_on_a_side']}")
    print(f"different document count under one id: {report['document_count_mismatch']}")
    print_deltas(report)
    print_gpu_time(report, before, after)
    if report["compared"]:
        print()
        print(f"max absolute score difference: {report['max_abs_score_diff']:.6g}")
        for cut in TOP_CUTS:
            share = report[f"top{cut}_changed_share"]
            print(f"top {cut} set differs: {share:.1%} of compared requests")
    return 0


# --- command line ------------------------------------------------------------


def env_override(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, value


def port_number(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be in the range 1 to 65535")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def positive_seconds(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value


def server_options(server_args_example: str) -> argparse.ArgumentParser:
    """The options every measuring mode takes, as a parent parser."""
    server = argparse.ArgumentParser(add_help=False)
    server.add_argument("--bin", required=True, help="server binary to start")
    server.add_argument(
        "--server-args",
        type=shlex.split,
        default=[],
        help=(
            "arguments for the server, as one string in the = form, for example "
            f'--server-args="{server_args_example}"; --host and --port are added'
        ),
    )
    server.add_argument(
        "--env",
        type=env_override,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "environment override for the server, repeatable; the result records "
            "every name, and the value only for MLXCEL_ and MLX_ names"
        ),
    )
    server.add_argument(
        "--port",
        type=port_number,
        default=18091,
        help="free port for the server to listen on, 1 to 65535 (default: 18091)",
    )
    server.add_argument(
        "--requests", required=True, help="private JSON Lines request file, outside the repository"
    )
    server.add_argument("--out", required=True, help="result JSON file to write")
    server.add_argument(
        "--commit", required=True, help="source revision the binary was built from"
    )
    server.add_argument(
        "--live-traffic",
        required=True,
        choices=("yes", "no", "unknown"),
        help="whether other traffic was using the same GPU during the run",
    )
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    server = server_options("-m <dir> --reranker-model <dir>")

    replay = modes.add_parser(
        "replay", parents=[server], help="send the requests one at a time"
    )
    replay.set_defaults(run=run_replay)

    concurrent = modes.add_parser(
        "concurrent", parents=[server], help="send the requests with N in flight"
    )
    concurrent.add_argument(
        "--in-flight",
        type=positive_int,
        required=True,
        help="requests kept in flight at once (the documented uses are 2 and 3)",
    )
    concurrent.set_defaults(run=run_replay)

    startup = modes.add_parser(
        "startup",
        parents=[server],
        help="time the first request, sent twice, on each of K fresh server starts",
    )
    startup.add_argument(
        "--starts", type=positive_int, required=True, help="number of fresh server starts"
    )
    startup.add_argument(
        "--page-cache",
        required=True,
        choices=("warm", "cold", "unknown"),
        help="state of the page cache for the model files before the first start",
    )
    startup.set_defaults(run=run_startup)

    abandon = modes.add_parser(
        "abandon",
        parents=[server],
        help=(
            "time the smallest request alone and right after hanging up on each "
            "large request"
        ),
    )
    abandon.add_argument(
        "--abandon-after",
        type=positive_seconds,
        required=True,
        metavar="SECONDS",
        help=(
            "how long a large request is left open before its connection is "
            "closed unread (the documented use is 1.0)"
        ),
    )
    abandon.add_argument(
        "--min-documents",
        type=positive_int,
        required=True,
        help=(
            "a request with at least this many documents is a large request "
            "(the documented use is 50)"
        ),
    )
    abandon.set_defaults(run=run_abandon)

    compare = modes.add_parser(
        "compare",
        help="compare two replay or concurrent result files, or two abandon result files",
    )
    compare.add_argument("--before", required=True, help="baseline result file")
    compare.add_argument("--after", required=True, help="result file being judged")
    compare.set_defaults(run=run_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (InputError, ServerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
