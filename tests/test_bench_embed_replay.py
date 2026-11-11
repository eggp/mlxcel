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

"""Tests for scripts/bench_embed_replay.py.

The server is a stub: a short Python program this file writes to the system
temp directory and the tool starts in place of mlxcel-server. It embeds
"synthetic text N" as the vector (N, 1, 0, -2) in base64, so every expected
vector is known. `ioreg` is the stub of tests/test_bench_rerank_replay.py,
whose test harness this file reuses, and the only `ioreg` on PATH. No model,
GPU or network beyond the loopback socket is involved.

Run with:
    python3 -m unittest tests/test_bench_embed_replay.py -v
"""

import base64
import json
import os
import pathlib
import platform
import struct
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import bench_embed_replay as ber  # noqa: E402
import bench_rerank_replay as brr  # noqa: E402
import test_bench_rerank_replay as rerank_tests  # noqa: E402

STUB_SERVER = r'''
import base64, json, os, socket, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

mode = sys.argv[1]
port = int(sys.argv[sys.argv.index("--port") + 1])
if mode == "exit-early":
    sys.exit(3)
if mode == "never-ready":
    time.sleep(600)
bias = float(os.environ.get("MLXCEL_STUB_BIAS", "0"))
barrier = threading.Barrier(int(mode.split("-")[1])) if mode.startswith("barrier-") else None
# "hold-N" serves one request at a time and never answers a request of N or
# more texts: that one ends when its caller hangs up. What it did is appended
# to the file MLXCEL_STUB_LOG names.
hold = int(mode.split("-")[1]) if mode.startswith("hold-") else None
worker = threading.Lock()
lock = threading.Lock()
served = [0]


def note(text):
    with open(os.environ["MLXCEL_STUB_LOG"], "a") as log:
        log.write(text + "\n")


def vector(text):
    values = [float(text.rsplit(" ", 1)[1]) + bias, 1.0, 0.0, -2.0]
    if mode == "nan-vector":
        values[1] = float("nan")
    return base64.b64encode(struct.pack("<4f", *values)).decode()


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
        texts = request["input"]
        if self.path != "/v1/embeddings" or request.get("encoding_format") != "base64":
            return self.reply(400, {"error": "the stub answers base64 embeddings only"})
        with lock:
            served[0] += 1
            nth = served[0]
        if mode == "dies-on-2" and nth == 2:
            os._exit(7)
        if mode == "status-500" or (mode == "second-fails" and nth > 1):
            return self.reply(500, {"error": texts[0]})
        if mode == "bad-response":
            return self.reply(200, {"data": [{"index": 0}]})
        if hold is not None:
            with worker:
                if len(texts) < hold:
                    note(f"reply {len(texts)}")
                else:
                    self.connection.settimeout(30)
                    try:
                        gone = self.connection.recv(1) == b""
                    except socket.timeout:
                        gone = False
                    except OSError:
                        gone = True
                    if gone:
                        return note(f"closed {len(texts)}")
                    note(f"still open {len(texts)}")
                    return self.reply(504, {"error": "the caller did not hang up"})
        if barrier is not None:
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                return self.reply(503, {"error": "not enough requests in flight"})
        data = [
            {"object": "embedding", "index": i, "embedding": vector(text)}
            for i, text in enumerate(texts)
        ]
        data.reverse()
        tokens = sum(len(text.split()) for text in texts)
        self.reply(200, {"object": "list", "data": data, "model": "stub",
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
        if mode == "dies-after-1":
            self.wfile.flush()
            os._exit(7)


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''

REQUESTS = [
    {"id": "alpha", "texts": ["synthetic text 3", "synthetic text 9", "synthetic text 1"]},
    {"texts": [f"synthetic text {n}" for n in range(20, 32)]},
    {"id": "gamma", "texts": ["synthetic text 40", "synthetic text 41"]},
    {"texts": ["synthetic text 5"]},
]
REQUEST_TEXTS = [text for r in REQUESTS for text in r["texts"]]
ON_MACOS = sys.platform == "darwin"
EARLIER = rerank_tests.EARLIER
MIB = 1024**2


def vector(*values) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def components(encoded: str) -> tuple:
    raw = base64.b64decode(encoded)
    return struct.unpack(f"<{len(raw) // 4}f", raw)


class ToolTestCase(rerank_tests.ToolTestCase):
    """The rerank tool's harness with this tool, its stub server and requests."""

    TOOL = ber
    STUB = STUB_SERVER
    REQUESTS = REQUESTS

    def assert_no_request_text(self, *texts):
        for text in REQUEST_TEXTS:
            for written in texts:
                self.assertNotIn(text, written)


class ReplayTest(ToolTestCase):
    def test_records_every_request_with_vectors_in_text_order(self):
        code, stdout, _, result = self.run_tool("replay")
        self.assertEqual(code, 0)
        rows = result["requests"]
        self.assertEqual([row["id"] for row in rows], ["alpha", 2, "gamma", 4])
        self.assertEqual([row["texts"] for row in rows], [3, 12, 2, 1])
        self.assertEqual([row["prompt_tokens"] for row in rows], [9, 36, 6, 3])
        self.assertEqual(
            [components(v) for v in rows[0]["vectors"]],
            [(3, 1, 0, -2), (9, 1, 0, -2), (1, 1, 0, -2)],
        )
        self.assertEqual([components(v)[0] for v in rows[1]["vectors"]], list(range(20, 32)))
        for row in rows:
            self.assertEqual(set(row), set(ber.REQUEST_ROW_KEYS))
            self.assertEqual(row["status"], 200)
            self.assertIsNone(row["error"])
            self.assertEqual(row["in_flight"], 1)
            self.assertGreater(row["wall_ms"], 0)
            self.assert_footprint(row["footprint_peak_bytes"])
            self.assert_footprint(row["footprint_retained_bytes"])
        self.assertIn("id=alpha texts=3 status=200", stdout)
        self.assertIn("4 of 4 requests embedded, 0 failed", stdout)

    def test_records_the_run_once_with_the_server_gpu_time(self):
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
        # The stub lists four clients of the server, one without `AppUsage`
        # and one with an empty one, and clients of other processes.
        self.assertEqual(run["gpu_time_ns"], rerank_tests.STUB_GPU_TIME_NS)
        self.assertEqual(run["gpu_time_source"], "ioreg")

    def test_unreadable_gpu_time_is_null_with_the_reason_and_the_run_completes(self):
        for how, source in rerank_tests.GPU_TIME_FAILURES.items():
            with self.subTest(how=how), self.ioreg_broken(how):
                code, _, _, result = self.run_tool("replay")
                self.assertEqual(code, 0)
                self.assertIsNone(result["run"]["gpu_time_ns"])
                self.assertEqual(result["run"]["gpu_time_source"], source)
                self.assertEqual(len(result["requests"]), 4)

    def test_environment_reaches_the_server_and_only_safe_values_are_recorded(self):
        code, stdout, stderr, result = self.run_tool(
            "replay", extra=["--env", "MLXCEL_STUB_BIAS=0.5", "--env", "STUB_TOKEN=hunter2"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(components(result["requests"][0]["vectors"][0]), (3.5, 1, 0, -2))
        self.assertEqual(result["run"]["env_names"], ["MLXCEL_STUB_BIAS", "STUB_TOKEN"])
        self.assertEqual(result["run"]["env_values"], {"MLXCEL_STUB_BIAS": "0.5"})
        for text in (self.out.read_text(encoding="utf-8"), stdout, stderr):
            self.assertNotIn("hunter2", text)

    def test_request_text_reaches_neither_the_result_file_nor_the_console(self):
        code, stdout, stderr, _ = self.run_tool("replay")
        self.assertEqual(code, 0)
        self.assert_no_request_text(self.out.read_text(encoding="utf-8"), stdout, stderr)

    def test_stops_the_server_it_started(self):
        self.run_tool("replay")
        self.assertEqual(len(self.started), 1)
        self.assertIsNotNone(self.started[0].poll())

    def test_non_200_is_recorded_without_the_response_body(self):
        # The stub echoes the first text in its error body.
        code, stdout, stderr, result = self.run_tool("replay", stub_mode="status-500")
        self.assertEqual(code, 1)
        for row in result["requests"]:
            self.assertEqual((row["status"], row["error"]), (500, "http_status"))
            self.assertIsNone(row["vectors"])
            self.assertIsNone(row["prompt_tokens"])
        self.assertIn("0 of 4 requests embedded, 4 failed", stdout)
        self.assert_no_request_text(self.out.read_text(encoding="utf-8"), stdout, stderr)

    def test_malformed_or_non_finite_200_response_is_a_failed_request(self):
        for stub_mode in ("bad-response", "nan-vector"):
            with self.subTest(stub_mode=stub_mode):
                code, _, _, _ = self.run_tool("replay", stub_mode=stub_mode)
                self.assertEqual(code, 1)
                result = json.loads(
                    self.out.read_text(encoding="utf-8"),
                    parse_constant=rerank_tests.reject_constant,
                )
                for row in result["requests"]:
                    self.assertEqual((row["status"], row["error"]), (200, "bad_response"))
                    self.assertIsNone(row["vectors"])


class ResponseShapeTest(unittest.TestCase):
    def test_vectors_are_put_back_in_text_order_with_the_token_count(self):
        payload = {
            "data": [
                {"index": 1, "embedding": vector(2.0)},
                {"index": 0, "embedding": vector(1.0)},
            ],
            "usage": {"prompt_tokens": 0},
        }
        self.assertEqual(
            ber.vectors_in_text_order(payload, 2), ([vector(1.0), vector(2.0)], 0)
        )

    def test_a_response_that_does_not_give_every_text_one_finite_vector_is_refused(self):
        ok = {"index": 0, "embedding": vector(1.0, 2.0)}
        usage = {"prompt_tokens": 3}

        def payload(*entries, usage=usage):
            return {"data": list(entries), "usage": usage}

        bad_payloads = {
            "not an object": [],
            "no data": {"usage": usage},
            "too few vectors": payload(ok),
            "repeated index": payload(ok, ok),
            "index out of range": payload(ok, {"index": 2, "embedding": vector(1.0)}),
            "float embedding": payload(ok, {"index": 1, "embedding": [1.0, 2.0]}),
            "not base64": payload(ok, {"index": 1, "embedding": "not base64!"}),
            "partial f32": payload(
                ok, {"index": 1, "embedding": base64.b64encode(b"\x00" * 6).decode()}
            ),
            "empty vector": payload(ok, {"index": 1, "embedding": ""}),
            "NaN component": payload(ok, {"index": 1, "embedding": vector(float("nan"))}),
            "infinite component": payload(ok, {"index": 1, "embedding": vector(float("inf"))}),
            "no usage": payload(ok, {"index": 1, "embedding": vector(1.0)}, usage=None),
            "boolean tokens": payload(
                ok, {"index": 1, "embedding": vector(1.0)}, usage={"prompt_tokens": True}
            ),
            "negative tokens": payload(
                ok, {"index": 1, "embedding": vector(1.0)}, usage={"prompt_tokens": -1}
            ),
        }
        for name, bad in bad_payloads.items():
            with self.subTest(name), self.assertRaises(ValueError):
                ber.vectors_in_text_order(bad, 2)


class ConcurrentTest(ToolTestCase):
    def test_keeps_the_given_number_of_requests_in_flight(self):
        # The stub answers a request only once two are in flight together.
        code, _, _, result = self.run_tool(
            "concurrent", stub_mode="barrier-2", extra=["--in-flight", "2"]
        )
        self.assertEqual(code, 0)
        self.assertEqual((result["run"]["mode"], result["run"]["in_flight"]), ("concurrent", 2))
        rows = result["requests"]
        self.assertEqual([row["status"] for row in rows], [200] * 4)
        self.assertEqual([row["in_flight"] for row in rows], [2] * 4)
        self.assertEqual(components(rows[2]["vectors"][1]), (41, 1, 0, -2))
        self.assertEqual(result["run"]["gpu_time_ns"], rerank_tests.STUB_GPU_TIME_NS)


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
        for row, proc in zip(result["starts"], self.started):
            self.assertEqual((row["id"], row["texts"]), ("alpha", 3))
            self.assertEqual((row["first_status"], row["first_error"]), (200, None))
            self.assertEqual((row["second_status"], row["second_error"]), (500, "http_status"))
            self.assertGreater(row["first_wall_ms"], 0)
            self.assert_footprint(row["footprint_before_bytes"])
            self.assert_footprint(row["footprint_after_first_bytes"])
            self.assertIsNotNone(proc.poll())
        self.assert_no_request_text(self.out.read_text(encoding="utf-8"), stdout, stderr)

    def test_two_successful_requests_per_start_exit_zero(self):
        code, _, _, result = self.run_tool(
            "startup", extra=["--starts", "1", "--page-cache", "warm"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(result["starts"][0]["second_status"], 200)


class AbandonTest(ToolTestCase):
    """REQUESTS has 3, 12, 2 and 1 texts: with `--min-texts 3` the large
    requests are "alpha" and line 2, and the probe is line 4."""

    def setUp(self):
        super().setUp()
        self.log = self.tmp / "stub.log"
        patcher = mock.patch.object(brr, "REQUEST_TIMEOUT_S", 60.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_abandon(self, stub_mode="hold-3", abandon_after="0.05", min_texts="3", **kwargs):
        extra = ["--abandon-after", abandon_after, "--min-texts", min_texts,
                 "--env", f"MLXCEL_STUB_LOG={self.log}"]
        return self.run_tool("abandon", stub_mode=stub_mode, extra=extra, **kwargs)

    def test_records_one_row_per_large_request_and_the_run(self):
        code, stdout, stderr, result = self.run_abandon()
        self.assertEqual(code, 0)
        rows = result["requests"]
        self.assertEqual([row["id"] for row in rows], ["alpha", 2])
        self.assertEqual([row["texts"] for row in rows], [3, 12])
        for row in rows:
            self.assertEqual(set(row), set(ber.ABANDON_ROW_KEYS))
            self.assertGreaterEqual(row["abandoned_after_ms"], 50.0)
            for probe in ("alone", "behind", "idle"):
                self.assertEqual(row[f"{probe}_status"], 200)
                self.assertIsNone(row[f"{probe}_error"])
        run = result["run"]
        self.assertEqual(
            (run["mode"], run["abandon_after_s"], run["min_texts"], run["probe_id"],
             run["probe_texts"], run["large_requests"], run["in_flight"]),
            ("abandon", 0.05, 3, 4, 1, 2, 1),
        )
        self.assertEqual(run["gpu_time_ns"], rerank_tests.STUB_GPU_TIME_NS)
        self.assertEqual(run["gpu_time_source"], "ioreg")
        self.assertIn("2 large requests: 2 abandoned, 0 not abandoned", stdout)
        self.assert_no_request_text(self.out.read_text(encoding="utf-8"), stdout, stderr)

    def test_server_sees_the_hang_up_and_is_idle_before_the_next_large_request(self):
        code, _, _, _ = self.run_abandon()
        self.assertEqual(code, 0)
        self.assertEqual(
            self.log.read_text(encoding="utf-8").split("\n")[:-1],
            ["reply 1", "closed 3", "reply 1", "reply 1"]
            + ["reply 1", "closed 12", "reply 1", "reply 1"],
        )

    def test_large_request_answered_before_the_delay_is_recorded_as_not_abandoned(self):
        code, stdout, _, result = self.run_abandon(stub_mode="ok", abandon_after="30")
        self.assertEqual(code, 0)
        self.assertEqual([row["abandoned_after_ms"] for row in result["requests"]], [None, None])
        self.assertIn("id=alpha texts=3 abandoned=no", stdout)

    def test_unreadable_gpu_time_is_null_with_the_reason_and_the_run_completes(self):
        for how, source in rerank_tests.GPU_TIME_FAILURES.items():
            with self.subTest(how=how), self.ioreg_broken(how):
                code, _, _, result = self.run_abandon(stub_mode="ok", abandon_after="30")
                self.assertEqual(code, 0)
                self.assertIsNone(result["run"]["gpu_time_ns"])
                self.assertEqual(result["run"]["gpu_time_source"], source)

    def refused_before_anything_runs(self, message, **kwargs):
        code, _, stderr, result = self.run_abandon(**kwargs)
        self.assertEqual(code, 2)
        self.assertIn(f"error: {self.requests}: {message}", stderr)
        self.assertIsNone(result)
        self.assertEqual(self.started, [])

    def test_request_files_without_a_usable_probe_and_large_request_are_refused(self):
        self.refused_before_anything_runs("no request has at least 50 texts", min_texts="50")
        self.refused_before_anything_runs(
            "the smallest request's text count is 1 and --min-texts is 1", min_texts="1"
        )
        self.requests = self.write_requests([REQUESTS[1]])
        self.refused_before_anything_runs("abandon needs at least two requests")


class SafetyTest(ToolTestCase):
    def test_occupied_port_is_refused_before_anything_is_executed_or_sent(self):
        executed = []
        popen = brr.subprocess.Popen

        def recording_popen(command, *args, **kwargs):
            executed.append(command)
            return popen(command, *args, **kwargs)

        for mode, extra in (
            ("replay", []),
            ("concurrent", ["--in-flight", "2"]),
            ("startup", ["--starts", "1", "--page-cache", "warm"]),
            ("abandon", ["--abandon-after", "0.05", "--min-texts", "3"]),
        ):
            with self.subTest(mode=mode), rerank_tests.OtherListener() as other, \
                    mock.patch.object(brr.subprocess, "Popen", side_effect=recording_popen):
                code, _, stderr, result = self.run_tool(mode, extra=extra, port=other.port)
                self.assertEqual(code, 2)
                self.assertIn(f"port {other.port}", stderr)
                self.assertIsNone(result)
                self.assertEqual(executed, [])
                self.assertEqual(other.posts, 0)

    def test_server_that_dies_during_a_run_fails_it_and_writes_no_result(self):
        for mode, stub_mode, extra in (
            ("replay", "dies-on-2", []),
            ("replay", "dies-after-1", []),
            ("concurrent", "dies-on-2", ["--in-flight", "2"]),
            ("startup", "dies-on-2", ["--starts", "2", "--page-cache", "warm"]),
            ("abandon", "dies-on-2", ["--abandon-after", "30", "--min-texts", "3"]),
        ):
            with self.subTest(mode=mode, stub_mode=stub_mode):
                self.out.write_text(EARLIER, encoding="utf-8")
                code, stdout, stderr, _ = self.run_tool(mode, stub_mode=stub_mode, extra=extra)
                self.assertEqual(code, 2)
                self.assertIn("server exited with code 7 during the run", stderr)
                self.assertEqual(self.out.read_text(encoding="utf-8"), EARLIER)
                self.assertNotIn("wrote", stdout)

    def test_failed_start_leaves_no_file_behind(self):
        out_dir = self.tmp / "out"
        out_dir.mkdir()
        code, _, _, _ = self.run_tool("replay", stub_mode="exit-early", out=out_dir / "r.json")
        self.assertEqual(code, 2)
        self.assertEqual(list(out_dir.iterdir()), [])

    def test_interrupted_writability_probe_leaves_no_file_behind(self):
        out_dir = self.tmp / "out"
        out_dir.mkdir()
        real_close = os.close

        def interrupted_close(handle):
            real_close(handle)
            raise KeyboardInterrupt

        with mock.patch.object(brr.os, "close", side_effect=interrupted_close), \
                self.assertRaises(KeyboardInterrupt):
            self.run_tool("replay", out=out_dir / "r.json")
        self.assertEqual(list(out_dir.iterdir()), [])
        self.assertEqual(self.started, [])

    def test_no_request_is_sent_once_a_worker_has_seen_the_server_dead(self):
        # The first worker stops between its liveness check and its send, the
        # server dies, and a second worker comes along. The first is let go
        # as soon as the second has either seen the death or is held back.
        log = []
        at_send, second_arrived, release = (threading.Event() for _ in range(3))
        proc = rerank_tests.FakeProcess(log, second_arrived)
        request = ber.EmbedRequest("a", ["synthetic text 1"])

        class Response:
            status = 200

            def read(self):
                return json.dumps({"data": [{"index": 0, "embedding": vector(1.0)}],
                                   "usage": {"prompt_tokens": 3}}).encode()

        class Connection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, method, path, **kwargs):
                log.append(path)
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
                ber.post(proc, 1, request)
            except brr.ServerError as exc:
                refused.append(exc)

        first, second = threading.Thread(target=worker), threading.Thread(target=worker)
        with mock.patch.object(brr, "SEND_LOCK", rerank_tests.SignallingLock(second_arrived)), \
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
        self.assertEqual(log[0], "/v1/embeddings")
        self.assertLess(log.index("send"), log.index("seen dead"), log)
        self.assertEqual(len(refused), 2)


class RequestFileTest(ToolTestCase):
    GOOD = json.dumps(REQUESTS[0])

    def test_malformed_lines_are_refused_by_line_number_without_their_content(self):
        texts = "`texts` must be a non-empty array of strings"
        cases = [
            ("not valid JSON", '{"texts": ["synthetic text 9"], '),
            ("not a JSON object", '["synthetic text 9"]'),
            (texts, '{"id": "x"}'),
            (texts, '{"texts": []}'),
            (texts, '{"texts": "synthetic text 9"}'),
            (texts, '{"texts": ["synthetic text 9", 3]}'),
            ("`id` must be a string", '{"id": 5, "texts": ["synthetic text 9"]}'),
            ("duplicate request id", self.GOOD),
        ]
        for message, bad_line in cases:
            with self.subTest(bad_line=bad_line):
                path = self.tmp / "bad.jsonl"
                path.write_text(f"{self.GOOD}\n{bad_line}\n", encoding="utf-8")
                with self.assertRaises(brr.InputError) as caught:
                    ber.load_requests(str(path))
                self.assertEqual(str(caught.exception), f"{path}:2: {message}")

    def test_bad_request_file_stops_the_run_before_a_server_starts(self):
        self.requests.write_text("synthetic text 9\n", encoding="utf-8")
        code, stdout, stderr, result = self.run_tool("replay")
        self.assertEqual(code, 2)
        self.assertIn(f"{self.requests}:1: not valid JSON", stderr)
        self.assertNotIn("synthetic text 9", stdout + stderr)
        self.assertIsNone(result)
        self.assertEqual(self.started, [])


def result_row(request_id, vectors, wall_ms=100.0, peak=1000 * MIB, retained=500 * MIB):
    return {
        "id": request_id,
        "texts": len(vectors) if vectors else 1,
        "in_flight": 1,
        "wall_ms": wall_ms,
        "footprint_peak_bytes": peak,
        "footprint_retained_bytes": retained,
        "status": 200 if vectors else 500,
        "error": None if vectors else "http_status",
        "prompt_tokens": 3 * len(vectors) if vectors else None,
        "vectors": vectors,
    }


def result_file(rows, mode="replay", gpu_time_ns=None, gpu_time_source="ioreg"):
    run = {"mode": mode, "commit": "abc12345", "date": "2026-01-01", "in_flight": 1,
           "live_traffic": "no", "gpu_time_ns": gpu_time_ns, "gpu_time_source": gpu_time_source}
    return {"run": run, "requests": rows}


A, B = vector(1.0, 2.0, 2.0, 0.0), vector(0.5, -0.25, 8.0, 1.0)
ZERO = vector(0.0, 0.0, 0.0, 0.0)
BEFORE = result_file(
    [
        result_row("same", [A, B], wall_ms=100.0),
        result_row("moved", [vector(3.0, 4.0, 0.0, 0.0)], wall_ms=100.0),
        result_row("zero", [ZERO], wall_ms=100.0),
        result_row("dimension", [A], wall_ms=100.0),
        result_row("fails-after", [A]),
        result_row("count", [A, A]),
        result_row("before-only", [A]),
    ],
    gpu_time_ns=3_000_000_000,
)
AFTER = result_file(
    [
        result_row("same", [A, B], wall_ms=90.0),
        result_row("moved", [vector(4.0, 3.0, 0.0, 0.0)], wall_ms=130.0),
        result_row("zero", [ZERO], wall_ms=100.0),
        result_row("dimension", [vector(1.0, 2.0, 2.0)], wall_ms=120.0),
        result_row("fails-after", None),
        result_row("count", [A]),
        result_row("after-only", [A]),
    ],
    gpu_time_ns=2_250_000_000,
)


class VectorComparisonTest(unittest.TestCase):
    def test_known_vectors(self):
        self.assertEqual(
            ber.text_difference(vector(3.0, 4.0, 0.0, 0.0), vector(4.0, 3.0, 0.0, 0.0)),
            (False, 1.0, 0.96),
        )
        self.assertEqual(ber.text_difference(A, A), (True, 0.0, 1.0))
        self.assertEqual(ber.text_difference(ZERO, A), (False, 2.0, None))
        self.assertIsNone(ber.text_difference(A, vector(1.0, 2.0, 2.0)))

    def test_bit_identical_means_the_same_bytes_not_equal_values(self):
        self.assertEqual(ber.text_difference(vector(0.0), vector(-0.0))[:2], (False, 0.0))

    def test_counts_and_vector_figures_over_the_compared_requests(self):
        report = ber.compare_runs(BEFORE, AFTER)
        self.assertEqual(
            {name: report[name] for name in (
                "compared", "only_before", "only_after", "failed_on_a_side",
                "text_count_mismatch", "texts_compared", "texts_identical",
                "dimension_mismatch", "zero_vector")},
            {"compared": 4, "only_before": 1, "only_after": 1, "failed_on_a_side": 1,
             "text_count_mismatch": 1, "texts_compared": 4, "texts_identical": 3,
             "dimension_mismatch": 1, "zero_vector": 1},
        )
        self.assertEqual(report["max_abs_component_diff"], 1.0)
        self.assertEqual(report["min_cosine_similarity"], 0.96)
        wall = report["wall_ms"]
        self.assertEqual((wall["n"], wall["min"], wall["median"], wall["max"]), (4, -10, 10, 30))
        self.assertEqual(
            (report["gpu_time_before_ns"], report["gpu_time_after_ns"],
             report["gpu_time_diff_ns"]),
            (3_000_000_000, 2_250_000_000, -750_000_000),
        )

    def test_nothing_in_common_reports_no_vector_figures(self):
        report = ber.compare_runs(result_file([result_row("a", [A])]), result_file([]))
        self.assertEqual((report["compared"], report["texts_compared"]), (0, 0))
        self.assertIsNone(report["max_abs_component_diff"])
        self.assertIsNone(report["min_cosine_similarity"])


def abandon_row(request_id, behind, **kwargs):
    row = rerank_tests.abandon_row(request_id, behind, **kwargs)
    return {("texts" if name == "documents" else name): value for name, value in row.items()}


def abandon_file(rows):
    run = {"mode": "abandon", "commit": "abc12345", "date": "2026-01-01",
           "live_traffic": "no", "abandon_after_s": 1.0, "min_texts": 50,
           "probe_id": "probe", "probe_texts": 1, "large_requests": len(rows),
           "gpu_time_ns": 4_000_000, "gpu_time_source": "ioreg"}
    return {"run": run, "requests": rows}


class CompareTest(ToolTestCase):
    def write_json(self, name, data):
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def compare(self, before, after):
        return self.run_main(
            ["compare", "--before", str(self.write_json("before.json", before)),
             "--after", str(self.write_json("after.json", after))]
        )

    def test_command_prints_the_report(self):
        code, stdout, _, _ = self.compare(BEFORE, AFTER)
        self.assertEqual(code, 0)
        for line in (
            "commit=abc12345 date=2026-01-01 in_flight=1 live_traffic=no requests=7",
            "compared: 4 requests embedded in both files",
            "only in one file: 1 before, 1 after",
            "failed in at least one file: 1",
            "different text count under one id: 1",
            "wall time (ms, n=4): min -10.00",
            "before: 3000.00 ms (ioreg)\nafter: 2250.00 ms (ioreg)\n"
            "after minus before: -750.00 ms",
            "texts compared: 4, bit-identical: 3, different dimension: 1, "
            "all-zero vector on a side: 1",
            "max absolute component difference: 1",
            "lowest cosine similarity: 0.960000000",
        ):
            self.assertIn(line, stdout)

    def test_two_replays_of_the_stub_compare_end_to_end(self):
        first, second, third = (self.tmp / f"{n}.json" for n in ("first", "second", "third"))
        self.assertEqual(self.run_tool("replay", out=first)[0], 0)
        self.assertEqual(self.run_tool("replay", out=second)[0], 0)
        self.assertEqual(
            self.run_tool("concurrent", out=third,
                          extra=["--in-flight", "2", "--env", "MLXCEL_STUB_BIAS=0.5"])[0],
            0,
        )
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(second)]
        )
        self.assertEqual(code, 0)
        self.assertIn("texts compared: 18, bit-identical: 18", stdout)
        self.assertIn("lowest cosine similarity: 1.000000000", stdout)
        self.assertIn("after minus before: +0.00 ms", stdout)
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(third)]
        )
        self.assertEqual(code, 0)
        self.assertIn("texts compared: 18, bit-identical: 0", stdout)
        self.assertIn("max absolute component difference: 0.5", stdout)

    def test_two_abandon_runs_compare_like_the_rerank_tool_with_gpu_time(self):
        first, second = self.tmp / "first.json", self.tmp / "second.json"
        for out in (first, second):
            extra = ["--abandon-after", "0.05", "--min-texts", "3",
                     "--env", f"MLXCEL_STUB_LOG={self.tmp / 'stub.log'}"]
            self.assertEqual(self.run_tool("abandon", stub_mode="hold-3", extra=extra, out=out)[0], 0)
        code, stdout, _, _ = self.run_main(
            ["compare", "--before", str(first), "--after", str(second)]
        )
        self.assertEqual(code, 0)
        self.assertIn("abandon_after_s=0.05 min_texts=3 probe_texts=1", stdout)
        self.assertIn("compared: 2 large requests abandoned in both files", stdout)
        self.assertIn("probe wall time behind an abandoned request", stdout)
        self.assertIn("server GPU time over the whole run", stdout)
        self.assertIn("after minus before: +0.00 ms", stdout)

    def test_abandon_files_report_the_probe_figures(self):
        before = abandon_file([abandon_row("a", 3000.0, alone=50.0)])
        after = abandon_file([abandon_row("a", 60.0, alone=52.0)])
        code, stdout, _, _ = self.compare(before, after)
        self.assertEqual(code, 0)
        self.assertIn("after minus before (ms, n=1): min -2940.00", stdout)
        self.assertIn("after minus before (ms, n=1): min +2.00", stdout)

    def test_files_of_different_modes_are_refused(self):
        code, stdout, stderr, _ = self.compare(BEFORE, abandon_file([abandon_row("a", 60.0)]))
        self.assertEqual(code, 2)
        self.assertIn("abandon results compare only with other abandon results", stderr)
        self.assertEqual(stdout, "")

    def refused(self, data):
        code, _, stderr, _ = self.compare(BEFORE, data)
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(stderr.startswith(f"error: {self.tmp / 'after.json'}: "), stderr)
        return stderr

    def test_rows_without_a_key_the_tool_writes_are_refused(self):
        for vectors in ([A], None):
            full = result_row("a", vectors)
            self.assertEqual(set(full), set(ber.REQUEST_ROW_KEYS))
            for key in full:
                with self.subTest(key=key, vectors=vectors):
                    row = {name: value for name, value in full.items() if name != key}
                    self.assertIn(
                        f"request entry 1: `{key}` is missing", self.refused(result_file([row]))
                    )
        full = abandon_row("a", 60.0)
        self.assertEqual(set(full), set(ber.ABANDON_ROW_KEYS))
        row = {name: value for name, value in full.items() if name != "texts"}
        self.assertIn("request entry 1: `texts` is missing", self.refused(abandon_file([row])))

    def test_malformed_rows_are_refused_without_a_traceback(self):
        def row(**changes):
            return {**result_row("a", [A, B]), **changes}

        cases = {
            "empty vectors": row(vectors=[], texts=0),
            "vectors not a list": row(vectors=A),
            "float vector": row(vectors=[[1.0, 2.0], B]),
            "null vector": row(vectors=[None, B]),
            "not base64": row(vectors=["not base64!", B]),
            "partial f32": row(vectors=[base64.b64encode(b"\x00" * 6).decode(), B]),
            "NaN component": row(vectors=[vector(float("nan")), B]),
            "texts not the vector count": row(texts=3),
            "text count not an integer": row(texts="2"),
            "boolean tokens": row(prompt_tokens=True),
            "negative tokens": row(prompt_tokens=-1),
            "boolean id": row(id=True),
            "text wall time": row(wall_ms="fast"),
        }
        for name, bad_row in cases.items():
            with self.subTest(name):
                self.assertIn("request entry 1", self.refused(result_file([bad_row])))

    def test_startup_and_rerank_result_files_are_refused(self):
        for data in (
            {"run": {"mode": "startup"}, "starts": []},
            rerank_tests.result_file([rerank_tests.result_row("a", [0.1])]),
        ):
            with self.subTest(mode=data["run"]["mode"]):
                self.refused(data)


class HelpTest(rerank_tests.HelpTest):
    TOOL = ber
    OPTIONS = {
        **rerank_tests.HelpTest.OPTIONS,
        "abandon": rerank_tests.SERVER_OPTIONS | {"--abandon-after", "--min-texts"},
    }


if __name__ == "__main__":
    unittest.main()
