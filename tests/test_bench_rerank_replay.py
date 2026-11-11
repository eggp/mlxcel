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

"""Tests for scripts/bench_rerank_replay.py.

The server is a stub: a short Python program this file writes to the system
temp directory and the tool starts in place of mlxcel-server. It scores
"synthetic doc N" as N/100, so every expected score is known. `ioreg` is a
stub too, the only one on PATH, so the real command is never run. No model,
GPU or network beyond the loopback socket is involved.

Run with:
    python3 -m unittest tests/test_bench_rerank_replay.py -v
"""

import argparse
import contextlib
import functools
import io
import json
import math
import os
import pathlib
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_rerank_replay as brr  # noqa: E402

STUB_SERVER = r'''
import json, os, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

mode = sys.argv[1]
port = int(sys.argv[sys.argv.index("--port") + 1])
if mode == "exit-early":
    sys.exit(3)
if mode == "never-ready":
    time.sleep(600)
bias = float(os.environ.get("MLXCEL_STUB_BIAS", "0"))
barrier = threading.Barrier(int(mode.split("-")[1])) if mode.startswith("barrier-") else None
# "hold-N" serves one request at a time, as a single rerank worker does, and
# never answers a request of N or more documents: that one ends when its
# caller hangs up. What it did is appended to the file MLXCEL_STUB_LOG names.
hold = int(mode.split("-")[1]) if mode.startswith("hold-") else None
worker = threading.Lock()
lock = threading.Lock()
served = [0]
state = os.environ["STUB_IOREG_STATE"]


def note(text):
    with open(os.environ["MLXCEL_STUB_LOG"], "a") as log:
        log.write(text + "\n")


def put(name, text):
    """Replace a file of the stub state directory in one step, so a reader
    never sees it half written."""
    with open(os.path.join(state, name + ".tmp"), "w") as fh:
        fh.write(text)
    os.replace(os.path.join(state, name + ".tmp"), os.path.join(state, name))


def read_state(name):
    try:
        with open(os.path.join(state, name)) as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def busy_until_seen(nth):
    """Under "busy-footprint", hold request `nth` until a footprint reading
    has seen it in flight, see ToolTestCase.footprint_seen_while_busy."""
    put("busy", str(nth))
    deadline = time.monotonic() + 10
    while read_state("seen") != str(nth) and time.monotonic() < deadline:
        time.sleep(0.005)
    os.remove(os.path.join(state, "busy"))


# The stub `ioreg` reports GPU time in proportion to the requests counted here,
# written before any reply, so a reading after a response counts its request.
put("served", "0")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {"data": [{"id": "stub"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            served[0] += 1
            nth = served[0]
            put("served", str(nth))
        if mode == "dies-on-2" and nth == 2:
            os._exit(7)
        if mode == "busy-footprint":
            busy_until_seen(nth)
        if mode == "status-500" or (mode == "second-fails" and nth > 1):
            return self.reply(500, {"error": request["query"]})
        if mode == "bad-response":
            return self.reply(200, {"results": [{"index": 0}]})
        if hold is not None:
            documents = len(request["documents"])
            with worker:
                if documents < hold:
                    note(f"reply {documents}")
                else:
                    self.connection.settimeout(30)
                    try:
                        gone = self.connection.recv(1) == b""
                    except socket.timeout:
                        gone = False
                    except OSError:
                        gone = True
                    if gone:
                        return note(f"closed {documents}")
                    note(f"still open {documents}")
                    return self.reply(504, {"error": "the caller did not hang up"})
        if barrier is not None:
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                return self.reply(503, {"error": "not enough requests in flight"})
        results = [
            {"index": i, "relevance_score": int(doc.rsplit(" ", 1)[1]) / 100 + bias}
            for i, doc in enumerate(request["documents"])
        ]
        if mode == "nan-score":
            results[0]["relevance_score"] = float("nan")
        results.sort(key=lambda result: -result["relevance_score"])
        self.reply(200, {"model": "stub", "object": "list", "results": results})
        if mode == "dies-after-1":
            self.wfile.flush()
            os._exit(7)


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''

REQUESTS = [
    {
        "id": "alpha",
        "query": "synthetic query 1",
        "documents": ["synthetic doc 3", "synthetic doc 9", "synthetic doc 1"],
    },
    {
        "query": "synthetic query 2",
        "documents": [f"synthetic doc {n}" for n in range(20, 32)],
    },
    {
        "id": "gamma",
        "query": "synthetic query 3",
        "documents": ["synthetic doc 40", "synthetic doc 41"],
    },
    {"query": "synthetic query 4", "documents": ["synthetic doc 5"]},
]
REQUEST_TEXTS = [r["query"] for r in REQUESTS] + [d for r in REQUESTS for d in r["documents"]]
ON_MACOS = sys.platform == "darwin"

# Stands in for `ioreg -r -w 0 -c AGXDeviceUserClient`, in the layout the real
# command prints. In the directory STUB_IOREG_STATE names it reads the started
# server's pid from `server.pid`, what to do from `mode` ("ok" when absent,
# "no-client" or "fails"), and from `served` how many requests the stub server
# has received so far. Under "ok" the server owns four clients: one with two
# usage entries, one with an empty usage list, one without any, and one more
# with usage, 1500 ns per request served plus 7 in all, so a run records
# 1500 ns for each request sent between its two readings and no other figure.
# Clients of two other processes, one whose pid begins with the server's, are
# listed in every mode.
STUB_IOREG = r'''
import os, sys

here = os.environ["STUB_IOREG_STATE"]


def read(name, default):
    try:
        with open(os.path.join(here, name)) as fh:
            return fh.read().strip()
    except FileNotFoundError:
        return default


mode = read("mode", "ok")
if mode == "fails":
    sys.stderr.write("ioreg: stub failure\n")
    sys.exit(1)
served = int(read("served", "0"))
pid = int(read("server.pid", "0"))


def client(number, owner_pid, name, usage):
    lines = [
        "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x%x, !registered, "
        "!matched, active, busy 0, retain 5>" % (0x100000000 + number),
        "    {",
    ]
    if usage is not None:
        entries = ",".join(
            '{"API"="Metal","lastSubmittedTime"=1700816668322916,"accumulatedGPUTime"=%d}' % ns
            for ns in usage
        )
        lines.append('      "AppUsage" = (%s)' % entries)
    lines.append('      "IOUserClientCreator" = "pid %d, %s"' % (owner_pid, name))
    lines += ["    }", "    ", ""]
    return "\n".join(lines)


clients = [client(1, 1, "launchd", [10**12]), client(2, pid * 10 + 1, "other", [10**9])]
if mode == "ok":
    clients += [
        client(3, pid, "mlxcel-server", [1000 * served, 7]),
        client(4, pid, "mlxcel-server", []),
        client(5, pid, "mlxcel-server", None),
        client(6, pid, "mlxcel-server", [500 * served]),
    ]
print("\n".join(clients))
'''
STUB_GPU_TIME_NS_PER_REQUEST = 1500
# Footprints the fake reading of footprint_seen_while_busy returns.
FOOTPRINT_IDLE, FOOTPRINT_BUSY = 100 * 1024**2, 900 * 1024**2
# How the stub `ioreg` is broken, and the `gpu_time_source` a run then records.
GPU_TIME_FAILURES = {
    "missing": "unavailable: ioreg not found",
    "fails": "unavailable: ioreg failed",
    "no-client": "unavailable: no GPU client of the server process",
}
# Keeps the stub `ioreg` directory until the test process exits.
_KEEP = []


@functools.cache
def stub_ioreg_bin() -> pathlib.Path:
    """A directory in the system temp directory holding only the stub `ioreg`,
    written once per test process: the first run of a newly written
    executable took about half a second on macOS, later runs a few ms."""
    holder = tempfile.TemporaryDirectory()
    _KEEP.append(holder)
    directory = pathlib.Path(holder.name)
    ioreg = directory / "ioreg"
    ioreg.write_text(f"#!{sys.executable}\n{STUB_IOREG}", encoding="utf-8")
    ioreg.chmod(0o755)
    return directory


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ToolTestCase(unittest.TestCase):
    """Runs the tool's `main` against the stub server in a temp directory,
    with the stub `ioreg` alone on PATH."""

    TOOL = brr
    STUB = STUB_SERVER
    REQUESTS = REQUESTS

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.stub = self.tmp / "stub_server.py"
        self.stub.write_text(self.STUB, encoding="utf-8")
        self.requests = self.write_requests(self.REQUESTS)
        self.out = self.tmp / "result.json"
        self.ioreg_state = self.tmp / "ioreg"
        self.ioreg_state.mkdir()
        self.started = []
        real_popen = subprocess.Popen

        def recording_popen(command, *args, **kwargs):
            proc = real_popen(command, *args, **kwargs)
            if "--port" in command:
                self.started.append(proc)
                (self.ioreg_state / "server.pid").write_text(str(proc.pid), encoding="utf-8")
            return proc

        for patcher in (
            mock.patch.object(brr.subprocess, "Popen", side_effect=recording_popen),
            mock.patch.object(brr, "READY_TIMEOUT_S", 60.0),
            mock.patch.object(brr, "READY_POLL_S", 0.02),
            mock.patch.dict(
                os.environ,
                {"PATH": str(stub_ioreg_bin()), "STUB_IOREG_STATE": str(self.ioreg_state)},
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_requests(self, requests, name="requests.jsonl"):
        path = self.tmp / name
        path.write_text("".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8")
        return path

    @contextlib.contextmanager
    def ioreg_broken(self, how):
        """`ioreg` missing from PATH, exiting non-zero, or listing no client
        of the server, one key of GPU_TIME_FAILURES."""
        if how == "missing":
            empty = self.tmp / "empty-bin"
            empty.mkdir(exist_ok=True)
            with mock.patch.dict(os.environ, {"PATH": str(empty)}):
                yield
            return
        mode = self.ioreg_state / "mode"
        mode.write_text(how, encoding="utf-8")
        try:
            yield
        finally:
            mode.unlink()

    def footprint_seen_while_busy(self):
        """Fakes the footprint reading for the stub server's "busy-footprint"
        mode: FOOTPRINT_BUSY while the stub holds a request, which it does
        until a reading has seen it, and FOOTPRINT_IDLE at any other time. A
        footprint window therefore peaks at FOOTPRINT_BUSY only when it is
        open while the request is in flight."""

        def read_footprint(pid):
            try:
                nth = (self.ioreg_state / "busy").read_text(encoding="utf-8")
            except FileNotFoundError:
                return FOOTPRINT_IDLE
            (self.ioreg_state / "seen").write_text(nth, encoding="utf-8")
            return FOOTPRINT_BUSY

        return mock.patch.object(brr, "read_footprint", side_effect=read_footprint)

    def run_tool(
        self, mode, stub_mode="ok", extra=(), out=None, binary=sys.executable, port=None
    ):
        """Returns (exit code, stdout, stderr, result or None)."""
        out = out or self.out
        argv = [
            mode,
            "--bin", binary,
            f"--server-args={shlex.quote(self.stub.as_posix())} {stub_mode}",
            "--port", str(port or free_port()),
            "--requests", str(self.requests),
            "--out", str(out),
            "--commit", "abc12345",
            "--live-traffic", "no",
            *extra,
        ]
        return self.run_main(argv, out)

    def run_main(self, argv, out=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = self.TOOL.main(argv)
        result = None
        if out is not None and out.is_file():
            result = json.loads(out.read_text(encoding="utf-8"))
        return code, stdout.getvalue(), stderr.getvalue(), result

    def assert_footprint(self, value):
        if ON_MACOS:
            self.assertIsInstance(value, int)
            self.assertGreater(value, 0)
        else:
            self.assertIsNone(value)


class ReplayTest(ToolTestCase):
    def test_records_every_request_with_scores_in_document_order(self):
        code, _, _, result = self.run_tool("replay")
        self.assertEqual(code, 0)
        rows = result["requests"]
        self.assertEqual([row["id"] for row in rows], ["alpha", 2, "gamma", 4])
        self.assertEqual([row["documents"] for row in rows], [3, 12, 2, 1])
        self.assertEqual(rows[0]["scores"], [0.03, 0.09, 0.01])
        self.assertEqual(rows[1]["scores"], [n / 100 for n in range(20, 32)])
        for row in rows:
            self.assertEqual(row["status"], 200)
            self.assertIsNone(row["error"])
            self.assertEqual(row["in_flight"], 1)
            self.assertGreater(row["wall_ms"], 0)
            self.assert_footprint(row["footprint_peak_bytes"])
            self.assert_footprint(row["footprint_retained_bytes"])
            if ON_MACOS:
                self.assertGreaterEqual(
                    row["footprint_peak_bytes"], row["footprint_retained_bytes"]
                )

    def test_records_the_run_once(self):
        code, _, _, result = self.run_tool("replay")
        self.assertEqual(code, 0)
        run = result["run"]
        self.assertEqual(run["mode"], "replay")
        self.assertEqual(run["in_flight"], 1)
        self.assertEqual(run["commit"], "abc12345")
        self.assertEqual(run["live_traffic"], "no")
        self.assertEqual(run["binary"], os.path.abspath(sys.executable))
        self.assertIn(platform.python_version(), run["binary_version"])
        self.assertEqual(run["server_args"], [self.stub.as_posix(), "ok"])
        self.assertEqual(run["date"], time.strftime("%Y-%m-%d"))
        self.assertEqual(
            run["footprint_source"], "proc_pid_rusage" if ON_MACOS else "unavailable"
        )
        self.assert_footprint(run["footprint_lifetime_max_bytes"])

    def test_footprint_peak_is_sampled_while_the_request_is_in_flight(self):
        with self.footprint_seen_while_busy():
            code, _, _, result = self.run_tool("replay", stub_mode="busy-footprint")
        self.assertEqual(code, 0)
        for row in result["requests"]:
            self.assertEqual(
                (row["footprint_peak_bytes"], row["footprint_retained_bytes"]),
                (FOOTPRINT_BUSY, FOOTPRINT_IDLE),
            )

    def test_environment_reaches_the_server_and_only_safe_values_are_recorded(self):
        code, stdout, stderr, result = self.run_tool(
            "replay", extra=["--env", "MLXCEL_STUB_BIAS=0.5", "--env", "STUB_TOKEN=hunter2"]
        )
        self.assertEqual(code, 0)
        for score, expected in zip(result["requests"][0]["scores"], [0.53, 0.59, 0.51]):
            self.assertAlmostEqual(score, expected)
        self.assertEqual(result["run"]["env_names"], ["MLXCEL_STUB_BIAS", "STUB_TOKEN"])
        self.assertEqual(result["run"]["env_values"], {"MLXCEL_STUB_BIAS": "0.5"})
        for text in (self.out.read_text(encoding="utf-8"), stdout, stderr):
            self.assertNotIn("hunter2", text)

    def test_request_text_reaches_neither_the_result_file_nor_the_console(self):
        code, stdout, stderr, _ = self.run_tool("replay")
        self.assertEqual(code, 0)
        written = self.out.read_text(encoding="utf-8")
        for text in REQUEST_TEXTS:
            self.assertNotIn(text, written)
            self.assertNotIn(text, stdout)
            self.assertNotIn(text, stderr)

    def test_stops_the_server_it_started(self):
        self.run_tool("replay")
        self.assertEqual(len(self.started), 1)
        self.assertIsNotNone(self.started[0].poll())

    def test_non_200_is_recorded_without_the_response_body(self):
        code, stdout, _, result = self.run_tool("replay", stub_mode="status-500")
        self.assertEqual(code, 1)
        for row in result["requests"]:
            self.assertEqual(row["status"], 500)
            self.assertEqual(row["error"], "http_status")
            self.assertIsNone(row["scores"])
        self.assertIn("0 of 4 requests scored, 4 failed", stdout)
        # The stub echoes the query in its error body.
        self.assertNotIn("synthetic query", self.out.read_text(encoding="utf-8") + stdout)

    def test_malformed_200_response_is_a_failed_request(self):
        code, _, _, result = self.run_tool("replay", stub_mode="bad-response")
        self.assertEqual(code, 1)
        self.assertEqual(result["requests"][0]["status"], 200)
        self.assertEqual(result["requests"][0]["error"], "bad_response")
        self.assertIsNone(result["requests"][0]["scores"])

    def test_server_that_never_becomes_ready_is_stopped_and_reported(self):
        with mock.patch.object(brr, "READY_TIMEOUT_S", 0.5):
            code, _, stderr, result = self.run_tool("replay", stub_mode="never-ready")
        self.assertEqual(code, 2)
        self.assertIn("did not become ready", stderr)
        self.assertIsNone(result)
        self.assertIsNotNone(self.started[0].poll())

    def test_server_that_exits_early_is_reported(self):
        code, _, stderr, result = self.run_tool("replay", stub_mode="exit-early")
        self.assertEqual(code, 2)
        self.assertIn("server exited early with code 3", stderr)
        self.assertIsNone(result)

    def test_binary_that_cannot_be_run_is_reported(self):
        code, _, stderr, _ = self.run_tool("replay", binary=str(self.tmp / "no-such-binary"))
        self.assertEqual(code, 2)
        self.assertIn("cannot run", stderr)

    def test_missing_output_directory_is_refused_before_the_server_starts(self):
        code, _, stderr, _ = self.run_tool("replay", out=self.tmp / "missing" / "r.json")
        self.assertEqual(code, 2)
        self.assertIn("directory does not exist", stderr)
        self.assertEqual(self.started, [])


# What an earlier run left at the output path.
EARLIER = '{"earlier": true}\n'


class OtherListener:
    """A listener that is not the started server. It answers like one, so a
    tool that mistook it for its own server would complete a run, and it
    counts what it was sent."""

    def __init__(self):
        self.posts = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply({"data": []})

            def do_POST(self):
                outer.posts += 1
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(
                    {
                        "results": [
                            {"index": i, "relevance_score": 0.5}
                            for i in range(len(request["documents"]))
                        ]
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class SignallingLock:
    """A lock that sets `waiting` when a thread finds it held."""

    def __init__(self, waiting):
        self._lock = threading.Lock()
        self._waiting = waiting

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self._waiting.set()
            self._lock.acquire()

    def __exit__(self, *exc_info):
        self._lock.release()


class FakeProcess:
    """Stands in for the started server: alive until `die`, and it logs every
    liveness check that finds it dead."""

    def __init__(self, log, seen_dead):
        self.returncode = None
        self._code = None
        self._log = log
        self._seen_dead = seen_dead

    def die(self, code):
        self._code = code

    def poll(self):
        if self._code is not None:
            self.returncode = self._code
            self._log.append("seen dead")
            self._seen_dead.set()
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            raise subprocess.TimeoutExpired("fake server", timeout)
        return self.returncode


class WrongProcessTest(ToolTestCase):
    def test_no_request_is_sent_once_a_worker_has_seen_the_server_dead(self):
        # The first worker stops between its liveness check and its send, the
        # server dies, and a second worker comes along. The first is let go
        # as soon as the second has either seen the death or is held back.
        log = []
        at_send, second_arrived, release = (threading.Event() for _ in range(3))
        proc = FakeProcess(log, second_arrived)
        request = brr.RerankRequest("a", "synthetic query 1", ["synthetic doc 1"])

        class Response:
            status = 200

            def read(self):
                return json.dumps({"results": [{"index": 0, "relevance_score": 0.5}]}).encode()

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                at_send.set()
                release.wait(timeout=30)
                log.append("send")

            def getresponse(self):
                return Response()

            def close(self):
                pass

        refused = []

        def worker():
            try:
                brr.post_to_running(proc, 1, request)
            except brr.ServerError as exc:
                refused.append(exc)

        first, second = threading.Thread(target=worker), threading.Thread(target=worker)
        with mock.patch.object(brr, "SEND_LOCK", SignallingLock(second_arrived)), \
                mock.patch.object(brr.http.client, "HTTPConnection", Connection):
            first.start()
            self.assertTrue(at_send.wait(timeout=30))
            proc.die(7)
            second.start()
            self.assertTrue(second_arrived.wait(timeout=30))
            release.set()
            first.join(timeout=30)
            second.join(timeout=30)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(log.count("send"), 1)
        self.assertLess(log.index("send"), log.index("seen dead"), log)
        self.assertEqual(len(refused), 2)

    def test_occupied_port_is_refused_before_the_binary_is_executed_at_all(self):
        executed = []
        popen = brr.subprocess.Popen

        def recording_popen(command, *args, **kwargs):
            executed.append(command)
            return popen(command, *args, **kwargs)

        with OtherListener() as other, mock.patch.object(
            brr.subprocess, "Popen", side_effect=recording_popen
        ):
            code, _, stderr, result = self.run_tool("replay", port=other.port)
        self.assertEqual(code, 2)
        self.assertIn(f"port {other.port}", stderr)
        self.assertIsNone(result)
        self.assertEqual(executed, [])

    def test_occupied_port_is_refused_before_anything_is_started_or_sent(self):
        for mode, extra in (
            ("replay", []),
            ("startup", ["--starts", "1", "--page-cache", "warm"]),
            ("abandon", ["--abandon-after", "0.05", "--min-documents", "3"]),
        ):
            with self.subTest(mode=mode), OtherListener() as other:
                code, _, stderr, result = self.run_tool(mode, extra=extra, port=other.port)
                self.assertEqual(code, 2)
                self.assertIn(f"port {other.port}", stderr)
                self.assertIsNone(result)
                self.assertEqual(self.started, [])
                self.assertEqual(other.posts, 0)

    def test_server_that_dies_on_a_request_fails_the_run_and_writes_no_result(self):
        self.out.write_text(EARLIER, encoding="utf-8")
        code, stdout, stderr, _ = self.run_tool("replay", stub_mode="dies-on-2")
        self.assertEqual(code, 2)
        self.assertIn("server exited with code 7 during the run", stderr)
        self.assertEqual(self.out.read_text(encoding="utf-8"), EARLIER)
        # Nothing is sent or reported after the death is known.
        self.assertIn("id=alpha", stdout)
        self.assertNotIn("id=gamma", stdout)
        self.assertNotIn("requests scored", stdout)
        self.assertNotIn("wrote", stdout)

    def test_server_that_dies_after_a_response_fails_the_run(self):
        code, stdout, stderr, result = self.run_tool("replay", stub_mode="dies-after-1")
        self.assertEqual(code, 2)
        self.assertIn("server exited with code 7 during the run", stderr)
        self.assertIsNone(result)
        self.assertNotIn("requests scored", stdout)

    def test_server_that_dies_under_concurrent_requests_fails_the_run(self):
        code, _, stderr, result = self.run_tool(
            "concurrent", stub_mode="dies-on-2", extra=["--in-flight", "2"]
        )
        self.assertEqual(code, 2)
        self.assertIn("server exited with code 7 during the run", stderr)
        self.assertIsNone(result)

    def test_server_that_dies_during_startup_mode_fails_the_run(self):
        code, _, stderr, result = self.run_tool(
            "startup", stub_mode="dies-on-2", extra=["--starts", "2", "--page-cache", "warm"]
        )
        self.assertEqual(code, 2)
        self.assertIn("server exited with code 7 during the run", stderr)
        self.assertIsNone(result)
        self.assertEqual(len(self.started), 1)


class OutputPathTest(ToolTestCase):
    def test_output_path_that_is_a_directory_is_refused_before_the_server_starts(self):
        code, _, stderr, _ = self.run_tool("replay", out=self.tmp)
        self.assertEqual(code, 2)
        self.assertIn(f"{self.tmp}: cannot write", stderr)
        self.assertEqual(self.started, [])

    @unittest.skipIf(os.geteuid() == 0, "root can write into a read-only directory")
    def test_read_only_directory_is_refused_before_the_server_starts(self):
        locked = self.tmp / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        self.addCleanup(locked.chmod, 0o755)
        code, _, stderr, _ = self.run_tool("replay", out=locked / "r.json")
        self.assertEqual(code, 2)
        self.assertIn(f"{locked / 'r.json'}: cannot write", stderr)
        self.assertEqual(self.started, [])

    @unittest.skipIf(os.geteuid() == 0, "root can write a read-only file")
    def test_read_only_existing_file_is_refused_before_the_server_starts(self):
        self.out.write_text(EARLIER, encoding="utf-8")
        self.out.chmod(0o444)
        code, _, stderr, _ = self.run_tool("replay")
        self.assertEqual(code, 2)
        self.assertIn(f"{self.out}: cannot write", stderr)
        self.assertEqual(self.started, [])
        self.assertEqual(self.out.read_text(encoding="utf-8"), EARLIER)

    def test_empty_output_path_is_refused_before_the_server_starts(self):
        # From a directory of its own, so that a probe beside the working
        # directory would still land in the temp directory.
        cwd = self.tmp / "cwd"
        cwd.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(cwd)
        code, _, stderr, _ = self.run_main(
            ["replay", "--bin", sys.executable,
             f"--server-args={shlex.quote(self.stub.as_posix())} ok",
             "--port", str(free_port()), "--requests", str(self.requests), "--out", "",
             "--commit", "abc12345", "--live-traffic", "no"]
        )
        self.assertEqual(code, 2)
        self.assertIn("--out", stderr)
        self.assertEqual(self.started, [])

    def test_interrupted_writability_probe_leaves_no_file_behind(self):
        out_dir = self.tmp / "out"
        out_dir.mkdir()
        real_close = os.close

        def interrupted_close(handle):
            real_close(handle)
            raise KeyboardInterrupt

        with mock.patch.object(brr.os, "close", side_effect=interrupted_close):
            with self.assertRaises(KeyboardInterrupt):
                brr.require_writable(str(out_dir / "r.json"))
        self.assertEqual(list(out_dir.iterdir()), [])

    def test_failed_run_leaves_no_file_behind(self):
        out_dir = self.tmp / "out"
        out_dir.mkdir()
        code, _, _, _ = self.run_tool("replay", stub_mode="exit-early", out=out_dir / "r.json")
        self.assertEqual(code, 2)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(list(out_dir.iterdir()), [])

    def test_failed_run_leaves_an_existing_file_untouched(self):
        self.out.write_text(EARLIER, encoding="utf-8")
        code, _, _, _ = self.run_tool("replay", stub_mode="exit-early")
        self.assertEqual(code, 2)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.out.read_text(encoding="utf-8"), EARLIER)

    def test_successful_run_replaces_an_existing_file(self):
        self.out.write_text(EARLIER, encoding="utf-8")
        code, _, _, result = self.run_tool("replay")
        self.assertEqual(code, 0)
        self.assertEqual(len(result["requests"]), 4)
        self.assertEqual([p.name for p in self.out.parent.iterdir() if p.suffix == ".json"],
                         ["result.json"])


def reject_constant(name):
    raise ValueError(f"not strict JSON: {name}")


class NonFiniteScoreTest(ToolTestCase):
    def test_non_finite_score_is_a_failed_request_and_the_file_is_strict_json(self):
        code, _, _, _ = self.run_tool("replay", stub_mode="nan-score")
        self.assertEqual(code, 1)
        result = json.loads(
            self.out.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        for row in result["requests"]:
            self.assertEqual(row["status"], 200)
            self.assertEqual(row["error"], "bad_response")
            self.assertIsNone(row["scores"])

    def test_non_finite_and_boolean_scores_are_refused(self):
        for raw in ("NaN", "Infinity", "-Infinity", "1e999", "1" + "0" * 400, "true", "false"):
            with self.subTest(score=raw):
                payload = json.loads('{"results": [{"index": 0, "relevance_score": %s}]}' % raw)
                with self.assertRaises(ValueError):
                    brr.scores_in_document_order(payload, 1)

    def test_result_writer_refuses_a_non_finite_number_and_writes_nothing(self):
        with self.assertRaises(ValueError):
            brr.write_result(str(self.out), {"run": {}, "requests": [{"wall_ms": math.nan}]})
        self.assertFalse(self.out.exists())


class PortOptionTest(ToolTestCase):
    def test_port_outside_1_to_65535_is_rejected_by_the_parser(self):
        for port in ("0", "-1", "65536", "70000"):
            with self.subTest(port=port):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    brr.build_parser().parse_args(
                        ["replay", "--bin", "b", "--requests", "r", "--out", "o",
                         "--commit", "c", "--live-traffic", "no", f"--port={port}"]
                    )
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--port", stderr.getvalue())
                self.assertIn("1 to 65535", stderr.getvalue())

    def test_ports_at_both_ends_of_the_range_are_accepted(self):
        for port in (1, 65535):
            args = brr.build_parser().parse_args(
                ["replay", "--bin", "b", "--requests", "r", "--out", "o",
                 "--commit", "c", "--live-traffic", "no", "--port", str(port)]
            )
            self.assertEqual(args.port, port)


class ConcurrentTest(ToolTestCase):
    def test_keeps_the_given_number_of_requests_in_flight(self):
        # The stub answers a request only once two are in flight together.
        code, _, _, result = self.run_tool(
            "concurrent", stub_mode="barrier-2", extra=["--in-flight", "2"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["run"]["mode"], "concurrent")
        self.assertEqual(result["run"]["in_flight"], 2)
        rows = result["requests"]
        self.assertEqual([row["id"] for row in rows], ["alpha", 2, "gamma", 4])
        self.assertEqual([row["status"] for row in rows], [200] * 4)
        self.assertEqual([row["in_flight"] for row in rows], [2] * 4)
        self.assertEqual(rows[0]["scores"], [0.03, 0.09, 0.01])


class StartupTest(ToolTestCase):
    def test_sends_the_first_request_twice_to_each_fresh_server(self):
        # The stub answers 200 to its first request and 500 to every later
        # one, so a 200 then a 500 per start means a fresh process each time.
        code, stdout, stderr, result = self.run_tool(
            "startup", stub_mode="second-fails", extra=["--starts", "2", "--page-cache", "cold"]
        )
        self.assertEqual(code, 1)
        self.assertEqual(result["run"]["mode"], "startup")
        self.assertEqual(result["run"]["page_cache"], "cold")
        self.assertEqual(len(self.started), 2)
        self.assertEqual([row["start"] for row in result["starts"]], [1, 2])
        for row, proc in zip(result["starts"], self.started):
            self.assertEqual(row["id"], "alpha")
            self.assertEqual(row["documents"], 3)
            self.assertEqual(row["first_status"], 200)
            self.assertEqual(row["second_status"], 500)
            self.assertIsNone(row["first_error"])
            self.assertEqual(row["second_error"], "http_status")
            self.assertGreater(row["first_wall_ms"], 0)
            self.assertGreater(row["second_wall_ms"], 0)
            self.assert_footprint(row["footprint_before_bytes"])
            self.assert_footprint(row["footprint_after_first_bytes"])
            self.assertIsNotNone(proc.poll())
        for text in REQUEST_TEXTS:
            self.assertNotIn(text, self.out.read_text(encoding="utf-8") + stdout + stderr)

    def test_two_successful_requests_per_start_exit_zero(self):
        code, _, _, result = self.run_tool(
            "startup", extra=["--starts", "1", "--page-cache", "warm"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["starts"][0]["second_status"], 200)
        self.assertIsNone(result["starts"][0]["first_error"])
        self.assertIsNone(result["starts"][0]["second_error"])

    def test_malformed_200_response_is_a_failed_start(self):
        for stub_mode in ("nan-score", "bad-response"):
            with self.subTest(stub_mode=stub_mode):
                code, _, _, _ = self.run_tool(
                    "startup", stub_mode=stub_mode, extra=["--starts", "1", "--page-cache", "warm"]
                )
                self.assertEqual(code, 1)
                row = json.loads(
                    self.out.read_text(encoding="utf-8"), parse_constant=reject_constant
                )["starts"][0]
                self.assertEqual((row["first_status"], row["second_status"]), (200, 200))
                self.assertEqual(row["first_error"], "bad_response")
                self.assertEqual(row["second_error"], "bad_response")


class RequestFileTest(ToolTestCase):
    GOOD = json.dumps(REQUESTS[0])

    def test_malformed_lines_are_refused_by_line_number_without_their_content(self):
        documents = "`documents` must be a non-empty array of strings"
        cases = [
            ("not valid JSON", '{"query": "synthetic query 9", '),
            ("not a JSON object", '["synthetic query 9"]'),
            ("`query` must be a string", '{"query": 7, "documents": ["synthetic doc 9"]}'),
            (documents, '{"query": "synthetic query 9"}'),
            (documents, '{"query": "synthetic query 9", "documents": []}'),
            (documents, '{"query": "synthetic query 9", "documents": ["synthetic doc 9", 3]}'),
            (
                "`id` must be a string",
                '{"id": 5, "query": "synthetic query 9", "documents": ["synthetic doc 9"]}',
            ),
            ("duplicate request id", self.GOOD),
        ]
        for message, bad_line in cases:
            with self.subTest(bad_line=bad_line):
                path = self.tmp / "bad.jsonl"
                path.write_text(f"{self.GOOD}\n{bad_line}\n", encoding="utf-8")
                with self.assertRaises(brr.InputError) as caught:
                    brr.load_requests(str(path))
                self.assertEqual(str(caught.exception), f"{path}:2: {message}")

    def test_unreadable_or_empty_file_is_refused(self):
        with self.assertRaisesRegex(brr.InputError, "cannot read"):
            brr.load_requests(str(self.tmp / "absent.jsonl"))
        empty = self.tmp / "empty.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(brr.InputError, "no requests"):
            brr.load_requests(str(empty))

    def test_bad_request_file_stops_the_run_before_a_server_starts(self):
        self.requests.write_text("synthetic query 9\n", encoding="utf-8")
        code, stdout, stderr, result = self.run_tool("replay")
        self.assertEqual(code, 2)
        self.assertIn(f"{self.requests}:1: not valid JSON", stderr)
        self.assertNotIn("synthetic query 9", stdout + stderr)
        self.assertIsNone(result)
        self.assertEqual(self.started, [])


class ResponseShapeTest(unittest.TestCase):
    def test_scores_are_put_back_in_document_order(self):
        payload = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0},
            ]
        }
        self.assertEqual(brr.scores_in_document_order(payload, 3), [0.5, 0.0, 0.9])

    def test_a_response_that_does_not_score_every_document_once_is_refused(self):
        ok = {"index": 0, "relevance_score": 0.5}
        bad_payloads = [
            [],
            {},
            {"results": [ok]},
            {"results": [ok, ok]},
            {"results": [ok, {"index": 2, "relevance_score": 0.1}]},
            {"results": [ok, {"index": 1}]},
            {"results": [ok, {"index": 1, "relevance_score": "0.1"}]},
            {"results": [ok, "x"]},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                brr.scores_in_document_order(payload, 2)


class FootprintTest(unittest.TestCase):
    def test_own_process_footprint_is_read_on_macos_and_unavailable_elsewhere(self):
        footprint = brr.read_footprint(os.getpid())
        lifetime_max = brr.read_lifetime_max_footprint(os.getpid())
        if ON_MACOS:
            self.assertEqual(brr.FOOTPRINT_SOURCE, "proc_pid_rusage")
            self.assertGreater(footprint, 1024**2)
            self.assertGreater(lifetime_max, 1024**2)
        else:
            self.assertEqual(brr.FOOTPRINT_SOURCE, "unavailable")
            self.assertIsNone(footprint)
            self.assertIsNone(lifetime_max)

    def test_window_keeps_the_peak_of_its_samples_and_the_last_as_retained(self):
        sampler = brr.FootprintSampler(pid=0)
        with mock.patch.object(brr, "read_footprint", side_effect=[100, 900, 300, 5000]):
            with sampler.window() as window:
                sampler.sample()
            sampler.sample()
        self.assertEqual(window, {"peak": 900, "retained": 300})

    def test_running_sampler_keeps_sampling_between_the_ends_of_a_window(self):
        sampled_three_times = threading.Event()
        calls = []

        def fake_read(pid):
            calls.append(pid)
            if len(calls) >= 3:
                sampled_three_times.set()
            return len(calls)

        with mock.patch.object(brr, "read_footprint", side_effect=fake_read), \
                mock.patch.object(brr, "SAMPLE_INTERVAL_S", 0.001):
            with brr.FootprintSampler(pid=42):
                self.assertTrue(sampled_three_times.wait(timeout=30))
        self.assertEqual(set(calls), {42})

    def test_window_reports_none_when_the_footprint_cannot_be_read(self):
        sampler = brr.FootprintSampler(pid=0)
        with mock.patch.object(brr, "read_footprint", return_value=None):
            with sampler.window() as window:
                sampler.sample()
        self.assertEqual(window, {"peak": None, "retained": None})


MIB = 1024**2


def result_row(request_id, scores, wall_ms=100.0, peak=1000 * MIB, retained=500 * MIB):
    return {
        "id": request_id,
        "documents": len(scores) if scores else 0,
        "in_flight": 1,
        "wall_ms": wall_ms,
        "footprint_peak_bytes": peak,
        "footprint_retained_bytes": retained,
        "status": 200 if scores else 500,
        "error": None if scores else "http_status",
        "scores": scores,
    }


def result_file(rows, mode="replay"):
    run = {"mode": mode, "commit": "abc12345", "date": "2026-01-01", "in_flight": 1,
           "live_traffic": "no"}
    return {"run": run, "requests": rows}


def swapped(scores, i, j):
    out = list(scores)
    out[i], out[j] = out[j], out[i]
    return out


# Twelve documents, best first: index i scores (12 - i) / 100.
TWELVE = [(12 - i) / 100 for i in range(12)]

BEFORE = result_file(
    [
        result_row("top5", TWELVE),
        result_row("top10", TWELVE),
        result_row("short", [0.1, 0.2, 0.3]),
        result_row("same", TWELVE, retained=None),
        result_row("before-only", TWELVE),
        result_row("fails-after", TWELVE),
        result_row("resized", TWELVE),
    ]
)
AFTER = result_file(
    [
        # Documents 4 and 5 trade places: the top 5 set changes, the top 10 does not.
        result_row("top5", swapped(TWELVE, 4, 5), wall_ms=110.0, peak=1000 * MIB),
        # Documents 9 and 10 trade places: only the top 10 set changes.
        result_row("top10", swapped(TWELVE, 9, 10), wall_ms=90.0, peak=1002 * MIB,
                   retained=510 * MIB),
        # Fewer documents than either cut-off: every document is in both sets.
        result_row("short", [0.3, 0.2, 0.6], wall_ms=100.0, peak=1004 * MIB,
                   retained=480 * MIB),
        result_row("same", TWELVE, wall_ms=140.0, peak=1006 * MIB),
        result_row(7, TWELVE),
        result_row("after-only", TWELVE),
        result_row("fails-after", None),
        result_row("resized", TWELVE[:11]),
    ]
)


class CompareTest(ToolTestCase):
    def test_counts_what_is_compared_and_what_is_left_out(self):
        report = brr.compare_runs(BEFORE, AFTER)
        self.assertEqual(report["compared"], 4)
        self.assertEqual(report["only_before"], 1)
        self.assertEqual(report["only_after"], 2)
        self.assertEqual(report["failed_on_a_side"], 1)
        self.assertEqual(report["document_count_mismatch"], 1)

    def test_score_difference_and_ranking_changes(self):
        report = brr.compare_runs(BEFORE, AFTER)
        self.assertAlmostEqual(report["max_abs_score_diff"], 0.3)
        self.assertEqual(report["top5_changed_share"], 0.25)
        self.assertEqual(report["top10_changed_share"], 0.25)

    def test_difference_distributions_are_after_minus_before(self):
        report = brr.compare_runs(BEFORE, AFTER)
        wall = report["wall_ms"]
        self.assertEqual((wall["n"], wall["min"], wall["median"], wall["max"]), (4, -10, 5, 40))
        self.assertAlmostEqual(wall["p5"], -8.5)
        self.assertAlmostEqual(wall["p95"], 35.5)
        peak = report["footprint_peak_bytes"]
        self.assertEqual(
            (peak["n"], peak["min"], peak["median"], peak["max"]),
            (4, 0, 3 * MIB, 6 * MIB),
        )
        # "same" has no retained reading on one side and is left out of this one.
        retained = report["footprint_retained_bytes"]
        self.assertEqual(
            (retained["n"], retained["min"], retained["median"], retained["max"]),
            (3, -20 * MIB, 0, 10 * MIB),
        )

    def test_nothing_in_common_reports_no_numbers(self):
        report = brr.compare_runs(result_file([result_row("a", TWELVE)]), result_file([]))
        self.assertEqual(report["compared"], 0)
        self.assertEqual(report["only_before"], 1)
        self.assertIsNone(report["wall_ms"])
        self.assertIsNone(report["max_abs_score_diff"])
        self.assertIsNone(report["top5_changed_share"])

    def test_top_set_breaks_ties_by_document_index(self):
        self.assertEqual(brr.top_set([0.5, 0.9, 0.5, 0.5], 2), {1, 0})
        self.assertEqual(brr.top_set([0.1, 0.2], 5), {0, 1})

    def test_spread_of_a_single_value(self):
        self.assertEqual(
            brr.spread([7.0]), {"n": 1, "min": 7.0, "p5": 7.0, "median": 7.0, "p95": 7.0, "max": 7.0}
        )

    def write_json(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_command_prints_the_report(self):
        before = self.write_json("before.json", BEFORE)
        after = self.write_json("after.json", AFTER)
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(before), "--after", str(after)]
        )
        self.assertEqual(code, 0)
        self.assertIn("compared: 4 requests scored in both files", stdout)
        self.assertIn("only in one file: 1 before, 2 after", stdout)
        self.assertIn("failed in at least one file: 1", stdout)
        self.assertIn("different document count under one id: 1", stdout)
        self.assertIn(
            "wall time (ms, n=4): min -10.00  p5 -8.50  median +5.00  p95 +35.50  max +40.00",
            stdout,
        )
        self.assertIn("peak footprint (MiB, n=4): min +0.00", stdout)
        self.assertIn("retained footprint (MiB, n=3): min -20.00", stdout)
        self.assertIn("max absolute score difference: 0.3", stdout)
        self.assertIn("top 5 set differs: 25.0% of compared requests", stdout)
        self.assertIn("top 10 set differs: 25.0% of compared requests", stdout)

    def test_unusable_result_files_are_refused(self):
        good = self.write_json("good.json", BEFORE)
        bad_files = {
            "cannot read": self.tmp / "absent.json",
            "not valid JSON": self.tmp / "broken.json",
            "not a replay, concurrent or abandon result file": self.write_json(
                "startup.json", {"run": {"mode": "startup"}, "starts": []}
            ),
        }
        (self.tmp / "broken.json").write_text("{", encoding="utf-8")
        for message, bad in bad_files.items():
            with self.subTest(message=message):
                code, _, stderr, _ = self.run_main(
                    ["compare", "--before", str(good), "--after", str(bad)]
                )
                self.assertEqual(code, 2)
                self.assertIn(f"{bad}: {message}", stderr)

    def refused(self, data):
        """Compares `data` with a good file; returns the error line."""
        good = self.write_json("good.json", BEFORE)
        bad = self.write_json("bad.json", data)
        code, _, stderr, _ = self.run_main(
            ["compare", "--before", str(good), "--after", str(bad)]
        )
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(stderr.startswith(f"error: {bad}: "), stderr)
        return stderr

    def test_result_file_with_a_repeated_id_is_refused_with_a_count(self):
        rows = [result_row("twice", TWELVE), result_row("twice", TWELVE),
                result_row(3, TWELVE), result_row(3, None), result_row("once", TWELVE)]
        stderr = self.refused(result_file(rows))
        self.assertIn("2 ids appear more than once", stderr)
        self.assertNotIn("twice", stderr)

    def test_malformed_result_rows_are_refused_without_a_traceback(self):
        def row(**changes):
            return {**result_row("a", [0.1, 0.2]), **changes}

        cases = {
            "empty scores": row(scores=[], documents=0),
            "unhashable id": row(id=["a"]),
            "object id": row(id={"a": 1}),
            "float id": row(id=1.5),
            "boolean id": row(id=True),
            "null id": row(id=None),
            "scores not a list": row(scores="0.1"),
            "text score": row(scores=[0.1, "0.2"]),
            "boolean score": row(scores=[0.1, True]),
            "null score": row(scores=[0.1, None]),
            "documents not the score count": row(documents=3),
            "documents not an integer": row(documents="2"),
            "boolean documents": row(scores=[0.1], documents=True),
            "text wall time": row(wall_ms="fast"),
            "text footprint": row(footprint_peak_bytes="big"),
        }
        for name, bad_row in cases.items():
            with self.subTest(name):
                stderr = self.refused(result_file([bad_row]))
                self.assertIn("request entry 1", stderr)

    def test_result_row_without_a_key_the_tool_writes_is_refused(self):
        for scores in ([0.1, 0.2], None):
            full = result_row("a", scores)
            for key in full:
                with self.subTest(key=key, scores=scores):
                    row = {name: value for name, value in full.items() if name != key}
                    stderr = self.refused(result_file([row]))
                    self.assertIn(f"request entry 1: `{key}` is missing", stderr)

    def test_result_files_the_tool_writes_are_accepted(self):
        for mode, stub_mode, extra, code in (
            ("replay", "ok", [], 0),
            ("replay", "status-500", [], 1),
            ("concurrent", "ok", ["--in-flight", "2"], 0),
        ):
            with self.subTest(mode=mode, stub_mode=stub_mode):
                out = self.tmp / f"{mode}-{stub_mode}.json"
                self.assertEqual(
                    self.run_tool(mode, stub_mode=stub_mode, extra=extra, out=out)[0], code
                )
                self.assertEqual(len(brr.load_result(str(out))["requests"]), 4)

    def test_non_finite_numbers_in_a_result_file_are_refused(self):
        text = json.dumps(result_file([result_row("a", [0.1, 0.2])]))
        for field, raw in (("scores", "[0.1, NaN]"), ("scores", "[Infinity, 0.2]"),
                           ("scores", "[0.1, 1e999]"), ("wall_ms", "-Infinity")):
            with self.subTest(field=field, raw=raw):
                good = self.write_json("good.json", BEFORE)
                bad = self.tmp / "bad.json"
                data = json.loads(text)
                data["requests"][0][field] = "PLACEHOLDER"
                bad.write_text(json.dumps(data).replace('"PLACEHOLDER"', raw), encoding="utf-8")
                code, _, stderr, _ = self.run_main(
                    ["compare", "--before", str(good), "--after", str(bad)]
                )
                self.assertEqual(code, 2)
                self.assertNotIn("Traceback", stderr)
                self.assertIn(f"error: {bad}: request entry 1", stderr)

    def test_recorded_failures_are_accepted_by_compare(self):
        failed = self.write_json("failed.json", result_file([result_row("a", None)]))
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(failed), "--after", str(failed)]
        )
        self.assertEqual(code, 0)
        self.assertIn("failed in at least one file: 1", stdout)

    def test_two_replays_of_the_stub_compare_end_to_end(self):
        first, second = self.tmp / "first.json", self.tmp / "second.json"
        self.assertEqual(self.run_tool("replay", out=first)[0], 0)
        self.assertEqual(
            self.run_tool("replay", out=second, extra=["--env", "MLXCEL_STUB_BIAS=0.5"])[0], 0
        )
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(second)]
        )
        self.assertEqual(code, 0)
        self.assertIn("compared: 4 requests scored in both files", stdout)
        self.assertIn("max absolute score difference: 0.5", stdout)
        self.assertIn("top 5 set differs: 0.0% of compared requests", stdout)


def recording_connection(events):
    """An HTTPConnection that appends what happens to each POST to `events`,
    as ("send" | "read" | "close", document count)."""
    real = brr.http.client.HTTPConnection

    class RecordingConnection(real):
        posted = None

        def request(self, method, url, body=None, headers={}, **kwargs):
            if method == "POST":
                self.posted = len(json.loads(body)["documents"])
                events.append(("send", self.posted))
            return super().request(method, url, body=body, headers=headers, **kwargs)

        def getresponse(self):
            if self.posted is not None:
                events.append(("read", self.posted))
            return super().getresponse()

        def close(self):
            if self.posted is not None:
                events.append(("close", self.posted))
            super().close()

    return RecordingConnection


class AbandonTest(ToolTestCase):
    """REQUESTS has 3, 12, 2 and 1 documents: with `--min-documents 3` the
    large requests are "alpha" and line 2, and the probe is line 4."""

    def setUp(self):
        super().setUp()
        self.log = self.tmp / "stub.log"
        patcher = mock.patch.object(brr, "REQUEST_TIMEOUT_S", 60.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_abandon(self, stub_mode="hold-3", abandon_after="0.05", min_documents="3", **kwargs):
        extra = ["--abandon-after", abandon_after, "--min-documents", min_documents,
                 "--env", f"MLXCEL_STUB_LOG={self.log}"]
        return self.run_tool("abandon", stub_mode=stub_mode, extra=extra, **kwargs)

    def test_records_one_row_per_large_request(self):
        code, stdout, _, _ = self.run_abandon()
        self.assertEqual(code, 0)
        result = json.loads(
            self.out.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        rows = result["requests"]
        self.assertEqual([row["id"] for row in rows], ["alpha", 2])
        self.assertEqual([row["documents"] for row in rows], [3, 12])
        for row in rows:
            self.assertEqual(set(row), set(brr.ABANDON_ROW_KEYS))
            self.assertGreaterEqual(row["abandoned_after_ms"], 50.0)
            self.assertGreater(row["alone_wall_ms"], 0)
            self.assertGreater(row["behind_wall_ms"], 0)
            for probe in ("alone", "behind", "idle"):
                self.assertEqual(row[f"{probe}_status"], 200)
                self.assertIsNone(row[f"{probe}_error"])
            self.assert_footprint(row["footprint_retained_bytes"])
        self.assertIn("2 large requests: 2 abandoned, 0 not abandoned", stdout)
        self.assertIn("0 with a failed probe", stdout)

    def test_records_the_run_once(self):
        code, _, _, result = self.run_abandon()
        self.assertEqual(code, 0)
        run = result["run"]
        self.assertEqual(run["mode"], "abandon")
        self.assertEqual(run["abandon_after_s"], 0.05)
        self.assertEqual(run["min_documents"], 3)
        self.assertEqual(run["probe_id"], 4)
        self.assertEqual(run["probe_documents"], 1)
        self.assertEqual(run["large_requests"], 2)
        self.assertEqual(run["in_flight"], 1)
        self.assertEqual(run["commit"], "abc12345")
        self.assertEqual(run["live_traffic"], "no")
        self.assertEqual(run["binary"], os.path.abspath(sys.executable))
        self.assertEqual(run["date"], time.strftime("%Y-%m-%d"))
        self.assert_footprint(run["footprint_lifetime_max_bytes"])

    def test_probe_is_the_first_request_with_the_fewest_documents(self):
        self.requests = self.write_requests(
            [REQUESTS[1], {"id": "first-small", **REQUESTS[3]}, {"id": "second-small", **REQUESTS[3]}]
        )
        code, _, _, result = self.run_abandon(stub_mode="ok", abandon_after="30")
        self.assertEqual(code, 0)
        self.assertEqual(result["run"]["probe_id"], "first-small")
        self.assertEqual([row["id"] for row in result["requests"]], [1])

    def test_large_request_is_closed_unread_and_the_probe_is_sent_after_the_close(self):
        events = []
        with mock.patch.object(brr.http.client, "HTTPConnection", recording_connection(events)):
            code, _, _, _ = self.run_abandon()
        self.assertEqual(code, 0)
        probe = [("send", 1), ("read", 1)]
        self.assertEqual(
            # A probe's connection is closed by the response and by the tool.
            [event for event in events if event != ("close", 1)],
            probe + [("send", 3), ("close", 3)] + probe + probe
            + probe + [("send", 12), ("close", 12)] + probe + probe,
        )

    def test_server_sees_the_hang_up_and_is_idle_before_the_next_large_request(self):
        # Per large request the stub answers the probe alone, sees the large
        # request's caller hang up, then answers the probe behind it and one
        # more probe before the next large request arrives.
        code, _, _, _ = self.run_abandon()
        self.assertEqual(code, 0)
        self.assertEqual(
            self.log.read_text(encoding="utf-8").split("\n")[:-1],
            ["reply 1", "closed 3", "reply 1", "reply 1"]
            + ["reply 1", "closed 12", "reply 1", "reply 1"],
        )

    def test_large_request_answered_before_the_delay_is_recorded_as_not_abandoned(self):
        events = []
        with mock.patch.object(brr.http.client, "HTTPConnection", recording_connection(events)):
            code, stdout, _, result = self.run_abandon(stub_mode="ok", abandon_after="30")
        self.assertEqual(code, 0)
        for row in result["requests"]:
            self.assertIsNone(row["abandoned_after_ms"])
            self.assertEqual((row["alone_status"], row["behind_status"]), (200, 200))
        self.assertIn(("read", 3), events)
        self.assertIn(("read", 12), events)
        self.assertIn("2 large requests: 0 abandoned, 2 not abandoned", stdout)
        self.assertIn("id=alpha docs=3 abandoned=no", stdout)

    def refused_before_anything_runs(self, message, **kwargs):
        executed = []
        popen = brr.subprocess.Popen

        def recording_popen(command, *args, **popen_kwargs):
            executed.append(command)
            return popen(command, *args, **popen_kwargs)

        with mock.patch.object(brr.subprocess, "Popen", side_effect=recording_popen):
            code, _, stderr, result = self.run_abandon(**kwargs)
        self.assertEqual(code, 2)
        self.assertIn(f"error: {self.requests}: {message}", stderr)
        self.assertIsNone(result)
        self.assertEqual(executed, [])

    def test_request_file_without_a_large_request_is_refused(self):
        self.refused_before_anything_runs(
            "no request has at least 50 documents", min_documents="50"
        )

    def test_request_file_with_a_single_request_is_refused(self):
        self.requests = self.write_requests([REQUESTS[1]])
        self.refused_before_anything_runs("abandon needs at least two requests")

    def test_request_file_whose_smallest_request_is_a_large_one_is_refused(self):
        # With 3, 12, 2 and 1 documents and `--min-documents 1` the probe
        # would be a large request itself.
        self.refused_before_anything_runs(
            "the smallest request's document count is 1 and --min-documents is 1",
            min_documents="1",
        )
        self.requests = self.write_requests([REQUESTS[0], REQUESTS[1]])
        self.refused_before_anything_runs(
            "the smallest request's document count is 3 and --min-documents is 3"
        )

    def test_large_request_that_cannot_be_sent_fails_the_run_and_writes_no_result(self):
        sent = []
        send_rerank = brr.send_rerank

        def failing_send(port, request):
            sent.append(len(request.documents))
            if len(request.documents) >= 3:
                return None, "transport:ConnectionRefusedError"
            return send_rerank(port, request)

        with mock.patch.object(brr, "send_rerank", side_effect=failing_send):
            code, stdout, stderr, result = self.run_abandon()
        self.assertEqual(code, 2)
        self.assertIn(
            "error: the request to abandon could not be sent "
            "(transport:ConnectionRefusedError); no result file is written",
            stderr,
        )
        self.assertIsNone(result)
        # The probe alone, then the large request; nothing after the failure.
        self.assertEqual(sent, [1, 3])
        self.assertIsNotNone(self.started[0].poll())
        for text in REQUEST_TEXTS:
            self.assertNotIn(text, stdout)
            self.assertNotIn(text, stderr)

    def test_abandon_delay_that_is_not_a_positive_number_is_rejected_by_the_parser(self):
        for delay in ("0", "-1", "nan", "inf", "soon"):
            with self.subTest(delay=delay):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    brr.build_parser().parse_args(
                        ["abandon", "--bin", "b", "--requests", "r", "--out", "o",
                         "--commit", "c", "--live-traffic", "no", "--min-documents", "50",
                         f"--abandon-after={delay}"]
                    )
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--abandon-after", stderr.getvalue())

    def test_failed_probe_is_recorded_and_the_file_is_still_written(self):
        # The stub answers 200 to the first request, the probe alone, and 500
        # to every later one.
        code, stdout, _, result = self.run_abandon(stub_mode="second-fails", abandon_after="30")
        self.assertEqual(code, 1)
        first = result["requests"][0]
        self.assertEqual((first["alone_status"], first["alone_error"]), (200, None))
        self.assertEqual((first["behind_status"], first["behind_error"]), (500, "http_status"))
        self.assertEqual((first["idle_status"], first["idle_error"]), (500, "http_status"))
        self.assertGreater(first["behind_wall_ms"], 0)
        self.assertIn("2 with a failed probe", stdout)
        # The stub echoes the query in its error body.
        self.assertNotIn("synthetic query", self.out.read_text(encoding="utf-8") + stdout)
        self.assertEqual(len(brr.load_result(str(self.out))["requests"]), 2)

    def test_server_that_exits_during_the_scenario_fails_the_run_and_writes_no_result(self):
        # The second request the stub receives is the first large one.
        self.out.write_text(EARLIER, encoding="utf-8")
        code, stdout, stderr, _ = self.run_abandon(stub_mode="dies-on-2", abandon_after="30")
        self.assertEqual(code, 2)
        self.assertIn("server exited with code 7 during the run", stderr)
        self.assertEqual(self.out.read_text(encoding="utf-8"), EARLIER)
        self.assertNotIn("large requests:", stdout)
        self.assertNotIn("wrote", stdout)

    def test_request_text_reaches_neither_the_result_file_nor_the_console(self):
        code, stdout, stderr, _ = self.run_abandon()
        self.assertEqual(code, 0)
        written = self.out.read_text(encoding="utf-8")
        for text in REQUEST_TEXTS:
            self.assertNotIn(text, written)
            self.assertNotIn(text, stdout)
            self.assertNotIn(text, stderr)


def abandon_row(request_id, behind, alone=50.0, abandoned=1000.0, behind_error=None):
    return {
        "id": request_id,
        "documents": 150,
        "abandoned_after_ms": abandoned,
        "alone_wall_ms": alone,
        "alone_status": 200,
        "alone_error": None,
        "behind_wall_ms": behind,
        "behind_status": 500 if behind_error else 200,
        "behind_error": behind_error,
        "idle_status": 200,
        "idle_error": None,
        "footprint_retained_bytes": 500 * MIB,
    }


def abandon_file(rows):
    run = {"mode": "abandon", "commit": "abc12345", "date": "2026-01-01",
           "live_traffic": "no", "abandon_after_s": 1.0, "min_documents": 50,
           "probe_id": "probe", "probe_documents": 1, "large_requests": len(rows)}
    return {"run": run, "requests": rows}


ABANDON_BEFORE = abandon_file(
    [
        abandon_row("a", 3400.0, alone=50.0),
        abandon_row("b", 3000.0, alone=52.0),
        abandon_row("c", 2600.0, alone=48.0),
        abandon_row("before-only", 3000.0),
        abandon_row("early-before", 60.0, abandoned=None),
        abandon_row("fails-after", 3000.0),
        abandon_row("early-and-failed", 60.0, abandoned=None),
    ]
)
ABANDON_AFTER = abandon_file(
    [
        abandon_row("a", 60.0, alone=51.0),
        abandon_row("b", 55.0, alone=50.0),
        abandon_row("c", 70.0, alone=49.0),
        abandon_row("after-only", 60.0),
        abandon_row(7, 60.0),
        abandon_row("early-before", 60.0),
        abandon_row("fails-after", 5.0, behind_error="http_status"),
        abandon_row("early-and-failed", 5.0, behind_error="http_status"),
    ]
)


class AbandonCompareTest(ToolTestCase):
    def write_json(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def compare(self, before, after):
        return self.run_main(
            ["compare", "--before", str(self.write_json("before.json", before)),
             "--after", str(self.write_json("after.json", after))]
        )

    def test_counts_what_is_compared_and_what_is_left_out(self):
        report = brr.compare_abandon_runs(ABANDON_BEFORE, ABANDON_AFTER)
        self.assertEqual(report["compared"], 3)
        self.assertEqual(report["only_before"], 1)
        self.assertEqual(report["only_after"], 2)
        self.assertEqual(report["not_abandoned_on_a_side"], 1)
        # A request with a failed probe is counted here even when it was not
        # abandoned either.
        self.assertEqual(report["failed_on_a_side"], 2)

    def test_distributions_cover_the_compared_requests_only(self):
        report = brr.compare_abandon_runs(ABANDON_BEFORE, ABANDON_AFTER)

        def ends(name):
            stats = report[name]
            return stats["n"], stats["min"], stats["median"], stats["max"]

        self.assertEqual(ends("behind_before"), (3, 2600, 3000, 3400))
        self.assertEqual(ends("behind_after"), (3, 55, 60, 70))
        self.assertEqual(ends("behind_diff"), (3, -3340, -2945, -2530))
        self.assertEqual(ends("alone_before"), (3, 48, 50, 52))
        self.assertEqual(ends("alone_after"), (3, 49, 50, 51))
        self.assertEqual(ends("alone_diff"), (3, -2, 1, 1))
        self.assertAlmostEqual(report["behind_diff"]["p5"], -3300.5)
        self.assertAlmostEqual(report["behind_diff"]["p95"], -2571.5)

    def test_nothing_in_common_reports_no_numbers(self):
        report = brr.compare_abandon_runs(ABANDON_BEFORE, abandon_file([]))
        self.assertEqual(report["compared"], 0)
        self.assertEqual(report["only_before"], 7)
        self.assertIsNone(report["behind_diff"])

    def test_command_prints_the_report(self):
        code, stdout, _, _ = self.compare(ABANDON_BEFORE, ABANDON_AFTER)
        self.assertEqual(code, 0)
        self.assertIn("abandon_after_s=1.0 min_documents=50 probe_documents=1", stdout)
        self.assertIn("compared: 3 large requests abandoned in both files", stdout)
        self.assertIn("only in one file: 1 before, 2 after", stdout)
        self.assertIn("not abandoned in at least one file: 1", stdout)
        self.assertIn("failed probe in at least one file: 2", stdout)
        self.assertIn(
            "before (ms, n=3): min 2600.00  p5 2640.00  median 3000.00  p95 3360.00  max 3400.00",
            stdout,
        )
        self.assertIn("after (ms, n=3): min 55.00", stdout)
        self.assertIn(
            "after minus before (ms, n=3): min -3340.00  p5 -3300.50  median -2945.00"
            "  p95 -2571.50  max -2530.00",
            stdout,
        )
        self.assertIn("after minus before (ms, n=3): min -2.00  p5 -1.70  median +1.00", stdout)

    def test_files_of_different_modes_are_refused(self):
        for before, after in ((ABANDON_BEFORE, BEFORE), (BEFORE, ABANDON_AFTER)):
            with self.subTest(before=before["run"]["mode"]):
                code, stdout, stderr, _ = self.compare(before, after)
                self.assertEqual(code, 2)
                self.assertIn(
                    f"holds {before['run']['mode']} results and {self.tmp / 'after.json'} "
                    f"holds {after['run']['mode']} results; abandon results compare only "
                    "with other abandon results",
                    stderr,
                )
                self.assertEqual(stdout, "")

    def refused(self, row):
        code, _, stderr, _ = self.compare(ABANDON_BEFORE, abandon_file([row]))
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(stderr.startswith(f"error: {self.tmp / 'after.json'}: "), stderr)
        return stderr

    def test_abandon_row_without_a_key_the_tool_writes_is_refused(self):
        full = abandon_row("a", 60.0)
        self.assertEqual(set(full), set(brr.ABANDON_ROW_KEYS))
        for key in full:
            with self.subTest(key=key):
                row = {name: value for name, value in full.items() if name != key}
                self.assertIn(f"request entry 1: `{key}` is missing", self.refused(row))

    def test_malformed_abandon_rows_are_refused_without_a_traceback(self):
        cases = {
            "boolean id": {"id": True},
            "list id": {"id": ["a"]},
            "text documents": {"documents": "150"},
            "text delay": {"abandoned_after_ms": "1000"},
            "boolean delay": {"abandoned_after_ms": True},
            "text wall time": {"behind_wall_ms": "slow"},
            "text footprint": {"footprint_retained_bytes": "big"},
            "error that is not text": {"alone_error": 1},
            "answered probe without a wall time": {"behind_wall_ms": None},
            "answered probe without a wall time, with an error": {
                "alone_wall_ms": None, "alone_error": "bad_response"
            },
            "no wall time and no error": {"alone_wall_ms": None, "alone_status": None},
        }
        for name, changes in cases.items():
            with self.subTest(name):
                self.assertIn("request entry 1", self.refused({**abandon_row("a", 60.0), **changes}))

    def test_failed_probe_rows_are_accepted_and_counted(self):
        failed = [
            abandon_row("a", 5.0, behind_error="http_status"),
            {**abandon_row("b", 5.0), "behind_status": None,
             "behind_error": "transport:ConnectionResetError"},
            {**abandon_row("c", None), "behind_status": None,
             "behind_error": "transport:ConnectionResetError"},
        ]
        code, stdout, _, _ = self.compare(ABANDON_BEFORE, abandon_file(failed))
        self.assertEqual(code, 0)
        self.assertIn("compared: 0 large requests", stdout)
        self.assertIn("failed probe in at least one file: 3", stdout)

    def test_replay_rows_under_the_abandon_mode_are_refused(self):
        self.assertIn("request entry 1", self.refused(result_row("a", TWELVE)))

    def test_abandon_file_with_a_repeated_id_is_refused(self):
        code, _, stderr, _ = self.compare(
            ABANDON_BEFORE, abandon_file([abandon_row("twice", 60.0), abandon_row("twice", 60.0)])
        )
        self.assertEqual(code, 2)
        self.assertIn("1 ids appear more than once", stderr)

    def test_two_abandon_runs_of_the_stub_compare_end_to_end(self):
        first, second = self.tmp / "first.json", self.tmp / "second.json"
        for out in (first, second):
            extra = ["--abandon-after", "0.05", "--min-documents", "3",
                     "--env", f"MLXCEL_STUB_LOG={self.tmp / 'stub.log'}"]
            self.assertEqual(
                self.run_tool("abandon", stub_mode="hold-3", extra=extra, out=out)[0], 0
            )
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(second)]
        )
        self.assertEqual(code, 0)
        self.assertIn("compared: 2 large requests abandoned in both files", stdout)
        self.assertIn("failed probe in at least one file: 0", stdout)


def gpu_client(owner, usage):
    """One client entry as `ioreg -r -w 0 -c AGXDeviceUserClient` prints it;
    `usage` None leaves out the `AppUsage` line."""
    lines = [
        "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1000011aa, "
        "!registered, !matched, active, busy 0, retain 5>",
        "    {",
    ]
    if usage is not None:
        entries = ",".join(
            '{"API"="Metal","lastSubmittedTime"=1700816668322916,"accumulatedGPUTime"=%d}' % ns
            for ns in usage
        )
        lines.append(f'      "AppUsage" = ({entries})')
    lines += [f'      "IOUserClientCreator" = "{owner}"', "    }", "    ", ""]
    return "\n".join(lines)


GPU_LISTING = "\n".join(
    [
        gpu_client("pid 4242, mlxcel-server", [129172208083, 17]),
        gpu_client("pid 4242, mlxcel-server", None),
        gpu_client("pid 4242, mlxcel-server", []),
        gpu_client("pid 42420, other", [5000]),
        gpu_client("pid 4242, mlxcel-server", [100]),
        gpu_client("pid 1, launchd", [7]),
    ]
)


class GpuTimeTest(ToolTestCase):
    def test_sums_every_usage_entry_over_the_clients_of_one_pid(self):
        self.assertEqual(brr.gpu_time_in(GPU_LISTING, 4242), 129172208083 + 17 + 100)
        self.assertEqual(brr.gpu_time_in(GPU_LISTING, 42420), 5000)

    def test_clients_without_usage_count_as_zero_and_a_pid_without_clients_as_none(self):
        no_usage = gpu_client("pid 4242, mlxcel-server", None) + gpu_client(
            "pid 4242, mlxcel-server", []
        )
        self.assertEqual(brr.gpu_time_in(no_usage, 4242), 0)
        self.assertIsNone(brr.gpu_time_in(GPU_LISTING, 424))
        self.assertIsNone(brr.gpu_time_in("", 4242))

    def test_runs_record_the_server_gpu_time(self):
        # `abandon` sends four requests per large request: the probe alone,
        # the large request, the probe behind it and one more probe.
        for mode, extra, sent in (
            ("replay", [], 4),
            ("concurrent", ["--in-flight", "2"], 4),
            ("abandon", ["--abandon-after", "30", "--min-documents", "3"], 8),
        ):
            with self.subTest(mode=mode):
                code, _, _, result = self.run_tool(mode, extra=extra)
                self.assertEqual(code, 0)
                self.assertEqual(
                    result["run"]["gpu_time_ns"], sent * STUB_GPU_TIME_NS_PER_REQUEST
                )
                self.assertEqual(result["run"]["gpu_time_source"], "ioreg")

    def test_unreadable_gpu_time_is_null_with_the_reason_and_the_run_completes(self):
        for how, source in GPU_TIME_FAILURES.items():
            for mode, extra in (
                ("replay", []),
                ("abandon", ["--abandon-after", "30", "--min-documents", "3"]),
            ):
                with self.subTest(how=how, mode=mode), self.ioreg_broken(how):
                    code, _, _, result = self.run_tool(mode, extra=extra)
                    self.assertEqual(code, 0)
                    self.assertIsNone(result["run"]["gpu_time_ns"])
                    self.assertEqual(result["run"]["gpu_time_source"], source)
                    self.assertEqual(len(result["requests"]), 4 if mode == "replay" else 2)

    def write_json(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def compare(self, before, after):
        return self.run_main(
            ["compare", "--before", str(self.write_json("before.json", before)),
             "--after", str(self.write_json("after.json", after))]
        )

    @staticmethod
    def with_gpu_time(data, ns, source="ioreg"):
        return {**data, "run": {**data["run"], "gpu_time_ns": ns, "gpu_time_source": source}}

    def test_compare_reports_the_gpu_time_of_both_runs_and_the_difference(self):
        for before, after in ((BEFORE, AFTER), (ABANDON_BEFORE, ABANDON_AFTER)):
            with self.subTest(mode=before["run"]["mode"]):
                code, stdout, _, _ = self.compare(
                    self.with_gpu_time(before, 2_500_000_000),
                    self.with_gpu_time(after, 2_000_000_000),
                )
                self.assertEqual(code, 0)
                self.assertIn(
                    "server GPU time over the whole run\n"
                    "before: 2500.00 ms (ioreg)\n"
                    "after: 2000.00 ms (ioreg)\n"
                    "after minus before: -500.00 ms\n",
                    stdout,
                )
        report = brr.compare_runs(
            self.with_gpu_time(BEFORE, 2_500_000_000), self.with_gpu_time(AFTER, 2_000_000_000)
        )
        self.assertEqual(
            (report["gpu_time_before_ns"], report["gpu_time_after_ns"], report["gpu_time_diff_ns"]),
            (2_500_000_000, 2_000_000_000, -500_000_000),
        )

    def test_compare_without_a_gpu_time_on_a_side_reports_no_difference(self):
        code, stdout, _, _ = self.compare(
            self.with_gpu_time(BEFORE, None, "unavailable: ioreg not found"), AFTER
        )
        self.assertEqual(code, 0)
        self.assertIn("before: n/a (unavailable: ioreg not found)\n", stdout)
        self.assertIn("after: n/a (not recorded)\n", stdout)
        self.assertIn("after minus before: n/a\n", stdout)

    def test_two_replays_of_the_stub_compare_their_gpu_time(self):
        first, second = self.tmp / "first.json", self.tmp / "second.json"
        self.assertEqual(self.run_tool("replay", out=first)[0], 0)
        self.assertEqual(self.run_tool("replay", out=second)[0], 0)
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(second)]
        )
        self.assertEqual(code, 0)
        self.assertIn("after minus before: +0.00 ms\n", stdout)

    def test_result_file_with_a_malformed_gpu_time_is_refused(self):
        for ns, source, message in (
            ("1500", "ioreg", "`run.gpu_time_ns` must be null or an integer"),
            (1.5, "ioreg", "`run.gpu_time_ns` must be null or an integer"),
            (True, "ioreg", "`run.gpu_time_ns` must be null or an integer"),
            (1500, 7, "`run.gpu_time_source` must be a string"),
        ):
            with self.subTest(ns=ns, source=source):
                code, _, stderr, _ = self.compare(BEFORE, self.with_gpu_time(AFTER, ns, source))
                self.assertEqual(code, 2)
                self.assertIn(f"error: {self.tmp / 'after.json'}: {message}", stderr)


def long_options(parser) -> dict:
    """The long options of each mode of `parser`, `--help` left out."""
    modes = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {
        mode: {
            option
            for action in sub._actions
            for option in action.option_strings
            if option.startswith("--") and option != "--help"
        }
        for mode, sub in modes.choices.items()
    }


def help_text(tool, argv) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        try:
            tool.main(argv)
        except SystemExit as exc:
            if exc.code != 0:
                raise
    return stdout.getvalue()


SERVER_OPTIONS = {
    "--bin", "--server-args", "--env", "--port", "--requests", "--out", "--commit",
    "--live-traffic",
}


class HelpTest(unittest.TestCase):
    TOOL = brr
    OPTIONS = {
        "replay": SERVER_OPTIONS,
        "concurrent": SERVER_OPTIONS | {"--in-flight"},
        "startup": SERVER_OPTIONS | {"--starts", "--page-cache"},
        "abandon": SERVER_OPTIONS | {"--abandon-after", "--min-documents"},
        "compare": {"--before", "--after"},
    }

    def test_every_mode_is_in_the_help_and_every_long_option_in_its_mode_help(self):
        self.assertEqual(long_options(self.TOOL.build_parser()), self.OPTIONS)
        top = help_text(self.TOOL, ["--help"])
        for mode, options in self.OPTIONS.items():
            with self.subTest(mode=mode):
                self.assertIn(mode, top)
                text = help_text(self.TOOL, [mode, "--help"])
                for option in options:
                    self.assertIn(option, text)


if __name__ == "__main__":
    unittest.main()
