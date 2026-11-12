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
"""Replay recorded embedding requests against one server binary and compare runs.

Starts the given mlxcel-server binary, sends every request of a local JSON
Lines file to POST /v1/embeddings, and writes one JSON result file with, per
request, the client wall time, the peak and the retained physical footprint of
the server process, the HTTP status, the token count the server reports and
the vectors, and for the whole run the GPU time of the server process. A second
run against another binary or configuration gives a second file, and `compare`
reports how the two differ, vector changes included.

The request file is private. Its text is sent to the local server and nowhere
else: result files and console output carry ids, counts, timings, memory and
GPU figures and vectors only. The server's own stdout and stderr are discarded
for the same reason.

Each input line is one JSON object:

  {"id": "optional-string", "texts": ["...", "..."]}

Without `id` the 1-based line number identifies the request. Each line is sent
as {"input": texts, "encoding_format": "base64"}, and the result file keeps
every vector as the base64 string the server returned: little-endian f32,
exact to the bit, and far smaller than the same vector as a JSON array.

The server process and its port, the output path, the footprint and GPU time
readings and the result file are handled by bench_rerank_replay.py, the
reranker's counterpart of this tool, whose help describes them.

Modes:
  replay      one request at a time
  concurrent  the same with N requests in flight
  startup     K fresh server starts, the first request sent twice per start
  abandon     how long a small request waits behind a large one whose caller
              hung up
  compare     two replay or concurrent result files, or two abandon ones

Examples:
  scripts/bench_embed_replay.py replay --bin target/release/mlxcel-server \\
      --server-args="-m <embedding dir>" \\
      --requests ~/private/embed-requests.jsonl --commit 0ba1b50b \\
      --live-traffic no --out ~/private/embed-base.json

  scripts/bench_embed_replay.py compare --before ~/private/embed-base.json \\
      --after ~/private/embed-fix.json

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import functools
import math
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import bench_rerank_replay as base

# --- request file ------------------------------------------------------------


@dataclass(frozen=True)
class EmbedRequest:
    id: str | int
    texts: list[str]


def embed_request(item: dict, request_id: str | int, where: str) -> EmbedRequest:
    if not base.is_string_list(item.get("texts")):
        raise base.InputError(f"{where}: `texts` must be a non-empty array of strings")
    return EmbedRequest(request_id, item["texts"])


def load_requests(path: str) -> list[EmbedRequest]:
    return base.read_request_file(path, embed_request)


# --- one request -------------------------------------------------------------


def decode_vector(encoded: str) -> tuple[bytes, tuple[float, ...]]:
    """The bytes and the values of a base64 vector.

    Raises ValueError unless `encoded` is strict base64 of a whole, non-zero
    number of finite little-endian f32 values.
    """
    raw = base64.b64decode(encoded, validate=True)
    if not raw or len(raw) % 4:
        raise ValueError("not a whole number of f32 values")
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("a component is not finite")
    return raw, values


def vectors_in_text_order(payload, text_count: int) -> tuple[list[str], int]:
    """The base64 vectors of a /v1/embeddings response, indexed like the
    request's texts, and its `usage.prompt_tokens`.

    Raises ValueError unless every text has exactly one vector `decode_vector`
    accepts and the token count is a non-negative integer.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != text_count:
        raise ValueError("data does not match the texts")
    vectors: list[str | None] = [None] * text_count
    for entry in data:
        index = entry.get("index") if isinstance(entry, dict) else None
        embedding = entry.get("embedding") if isinstance(entry, dict) else None
        if (
            type(index) is not int
            or not 0 <= index < text_count
            or vectors[index] is not None
            or not isinstance(embedding, str)
        ):
            raise ValueError("malformed data entry")
        decode_vector(embedding)
        vectors[index] = embedding
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        raise ValueError("malformed usage")
    return vectors, tokens


def send_embeddings(port: int, request: EmbedRequest):
    return base.send_json(
        port, "/v1/embeddings", {"input": request.texts, "encoding_format": "base64"}
    )


def read_embeddings(conn, request: EmbedRequest):
    """Returns (status, (vectors, prompt_tokens) or None, error)."""
    return base.read_json(
        conn, lambda payload: vectors_in_text_order(payload, len(request.texts))
    )


post = functools.partial(base.post_to_running, send=send_embeddings, read=read_embeddings)


def measure_request(proc, port: int, request: EmbedRequest, sampler, in_flight: int) -> dict:
    with sampler.window() as window:
        wall_ms, status, embedded, error = post(proc, port, request)
    vectors, tokens = embedded if embedded is not None else (None, None)
    return {
        "id": request.id,
        "texts": len(request.texts),
        "in_flight": in_flight,
        "wall_ms": wall_ms,
        "footprint_peak_bytes": window["peak"],
        "footprint_retained_bytes": window["retained"],
        "status": status,
        "error": error,
        "prompt_tokens": tokens,
        "vectors": vectors,
    }


# --- modes -------------------------------------------------------------------

# SHORTCUT: each mode's sequence of shared steps mirrors the rerank tool's mode of the same name instead of one runner both tools share; replace it when a third replay tool is added.


def run_replay(args: argparse.Namespace) -> int:
    in_flight = getattr(args, "in_flight", 1)
    requests = load_requests(args.requests)
    run = base.run_record(args)
    run["in_flight"] = in_flight
    with base.running_server(args, base.answers_http) as proc:
        gpu_start = base.read_gpu_time(proc.pid)
        with base.FootprintSampler(proc.pid) as sampler:

            def one(request: EmbedRequest) -> dict:
                row = measure_request(proc, args.port, request, sampler, in_flight)
                print(
                    f"id={row['id']} texts={row['texts']} "
                    f"status={row['status']} wall_ms={row['wall_ms']:.1f} "
                    f"peak_mib={base.mib(row['footprint_peak_bytes'])} "
                    f"retained_mib={base.mib(row['footprint_retained_bytes'])}",
                    flush=True,
                )
                return row

            with ThreadPoolExecutor(max_workers=in_flight) as pool:
                rows = list(pool.map(one, requests))
        run.update(base.gpu_time_fields(gpu_start, base.read_gpu_time(proc.pid)))
        run["footprint_lifetime_max_bytes"] = base.read_lifetime_max_footprint(proc.pid)
    failed = sum(1 for row in rows if row["vectors"] is None)
    print(f"{len(rows) - failed} of {len(rows)} requests embedded, {failed} failed")
    base.write_result(args.out, {"run": run, "requests": rows})
    return 1 if failed else 0


def run_startup(args: argparse.Namespace) -> int:
    request = load_requests(args.requests)[0]
    run = base.run_record(args)
    run["page_cache"] = args.page_cache
    rows = []
    failed = 0
    for start in range(1, args.starts + 1):
        with base.running_server(args, base.accepts_connections) as proc:
            before = base.read_footprint(proc.pid)
            first_ms, first_status, first_embedded, first_error = post(proc, args.port, request)
            after = base.read_footprint(proc.pid)
            second_ms, second_status, second_embedded, second_error = post(
                proc, args.port, request
            )
        failed += first_embedded is None or second_embedded is None
        rows.append(
            {
                "start": start,
                "id": request.id,
                "texts": len(request.texts),
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
            f"before_mib={base.mib(before)} after_first_mib={base.mib(after)}",
            flush=True,
        )
    base.write_result(args.out, {"run": run, "starts": rows})
    return 1 if failed else 0


def run_abandon(args: argparse.Namespace) -> int:
    requests = load_requests(args.requests)
    if len(requests) < 2:
        raise base.InputError(
            f"{args.requests}: abandon needs at least two requests, "
            "a probe and a request to abandon"
        )
    large = [r for r in requests if len(r.texts) >= args.min_texts]
    if not large:
        raise base.InputError(
            f"{args.requests}: no request has at least {args.min_texts} texts"
        )
    # `min` keeps the first of several smallest requests.
    probe = min(requests, key=lambda request: len(request.texts))
    if len(probe.texts) >= args.min_texts:
        raise base.InputError(
            f"{args.requests}: the smallest request's text count is "
            f"{len(probe.texts)} and --min-texts is {args.min_texts}; "
            "the probe must be smaller than a large request"
        )
    run = base.run_record(args)
    run["in_flight"] = 1
    run["abandon_after_s"] = args.abandon_after
    run["min_texts"] = args.min_texts
    run["probe_id"] = probe.id
    run["probe_texts"] = len(probe.texts)
    run["large_requests"] = len(large)
    rows = []
    with base.running_server(args, base.answers_http) as proc:
        gpu_start = base.read_gpu_time(proc.pid)
        for request in large:
            alone_ms, alone_status, _, alone_error = post(proc, args.port, probe)
            abandoned_after_ms = base.abandon_request(
                proc, args.port, request, args.abandon_after, send_embeddings, read_embeddings
            )
            behind_ms, behind_status, _, behind_error = post(proc, args.port, probe)
            retained = base.read_footprint(proc.pid)
            # A server that kept embedding the abandoned request has finished
            # it once the probe behind it and one more probe have been
            # answered, if it serves embedding requests in arrival order.
            _, idle_status, _, idle_error = post(proc, args.port, probe)
            rows.append(
                {
                    "id": request.id,
                    "texts": len(request.texts),
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
                f"id={request.id} texts={len(request.texts)} {delay} "
                f"alone_ms={alone_ms:.1f} status={alone_status} "
                f"behind_ms={behind_ms:.1f} status={behind_status} "
                f"idle_status={idle_status} retained_mib={base.mib(retained)}",
                flush=True,
            )
        run.update(base.gpu_time_fields(gpu_start, base.read_gpu_time(proc.pid)))
        run["footprint_lifetime_max_bytes"] = base.read_lifetime_max_footprint(proc.pid)
    abandoned = sum(1 for row in rows if row["abandoned_after_ms"] is not None)
    failed = sum(1 for row in rows if base.probe_failed(row))
    print(
        f"{len(rows)} large requests: {abandoned} abandoned, "
        f"{len(rows) - abandoned} not abandoned, {failed} with a failed probe"
    )
    base.write_result(args.out, {"run": run, "requests": rows})
    return 1 if failed else 0


# --- compare -----------------------------------------------------------------


# The keys of a row `measure_request` writes.
REQUEST_ROW_KEYS = (
    "id",
    "texts",
    "in_flight",
    "wall_ms",
    "footprint_peak_bytes",
    "footprint_retained_bytes",
    "status",
    "error",
    "prompt_tokens",
    "vectors",
)
ABANDON_ROW_KEYS = ("id", "texts", *base.ABANDON_MEASURE_KEYS)


def row_defect(row) -> str | None:
    """What makes `row` unusable for a comparison, or None.

    A row has every key in REQUEST_ROW_KEYS. It is a recorded failure
    (`vectors` null) or carries one vector `decode_vector` accepts per text.
    """
    defect = base.entry_defect(
        row,
        REQUEST_ROW_KEYS,
        ("wall_ms", "footprint_peak_bytes", "footprint_retained_bytes"),
        "texts",
    )
    if defect:
        return defect
    tokens = row["prompt_tokens"]
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        return "`prompt_tokens` must be null or a non-negative integer"
    vectors = row["vectors"]
    if vectors is None:
        return None
    if not isinstance(vectors, list) or not vectors:
        return "`vectors` must be null or a non-empty array"
    for vector in vectors:
        try:
            decode_vector(vector)
        except (TypeError, ValueError):
            return "`vectors` must hold base64 vectors of finite little-endian f32 values"
    if row["texts"] != len(vectors):
        return "`texts` does not match the number of vectors"
    return None


# The check each comparable mode's rows must pass.
ROW_DEFECTS = {
    "replay": row_defect,
    "concurrent": row_defect,
    "abandon": functools.partial(base.abandon_row_defect, count="texts"),
}


def text_difference(before: str, after: str) -> tuple[bool, float, float | None] | None:
    """For two base64 vectors: whether they are bit-identical, the largest
    absolute component difference, and the cosine similarity, None when
    either vector is all zeros. None instead of all three when the two
    differ in dimension."""
    raw_b, b = decode_vector(before)
    raw_a, a = decode_vector(after)
    if len(b) != len(a):
        return None
    max_diff = max(abs(x - y) for x, y in zip(a, b))
    norms = math.fsum(x * x for x in b) * math.fsum(y * y for y in a)
    cosine = math.fsum(x * y for x, y in zip(b, a)) / math.sqrt(norms) if norms else None
    return raw_b == raw_a, max_diff, cosine


def compare_runs(before: dict, after: dict) -> dict:
    """Differences, after minus before, over the requests both runs embedded,
    and the vector differences text by text."""
    report, pairs = base.pair_rows(before, after, "vectors", "text_count_mismatch")
    texts = identical = dimension_mismatch = zero_vector = 0
    max_diff = min_cosine = None
    for b, a in pairs:
        for b_vector, a_vector in zip(b["vectors"], a["vectors"]):
            difference = text_difference(b_vector, a_vector)
            if difference is None:
                dimension_mismatch += 1
                continue
            texts += 1
            same, diff, cosine = difference
            identical += same
            max_diff = diff if max_diff is None else max(max_diff, diff)
            if cosine is None:
                zero_vector += 1
            else:
                min_cosine = cosine if min_cosine is None else min(min_cosine, cosine)
    report.update(
        {
            "texts_compared": texts,
            "texts_identical": identical,
            "dimension_mismatch": dimension_mismatch,
            "zero_vector": zero_vector,
            "max_abs_component_diff": max_diff,
            "min_cosine_similarity": min_cosine,
        }
    )
    report.update(base.compare_gpu_time(before, after))
    return report


def run_compare(args: argparse.Namespace) -> int:
    before, after, abandon = base.load_pair(args, ROW_DEFECTS)
    if abandon:
        base.print_abandon_comparison(args, before, after, "texts")
        return 0
    base.print_headers(args, before, after, base.REPLAY_HEADER_FIELDS)
    report = compare_runs(before, after)
    print()
    print(f"compared: {report['compared']} requests embedded in both files")
    print(
        f"only in one file: {report['only_before']} before, {report['only_after']} after"
    )
    print(f"failed in at least one file: {report['failed_on_a_side']}")
    print(f"different text count under one id: {report['text_count_mismatch']}")
    base.print_deltas(report)
    base.print_gpu_time(report, before, after)
    print()
    print(
        f"texts compared: {report['texts_compared']}, "
        f"bit-identical: {report['texts_identical']}, "
        f"different dimension: {report['dimension_mismatch']}, "
        f"all-zero vector on a side: {report['zero_vector']}"
    )
    if report["max_abs_component_diff"] is not None:
        print(f"max absolute component difference: {report['max_abs_component_diff']:.6g}")
    if report["min_cosine_similarity"] is not None:
        print(f"lowest cosine similarity: {report['min_cosine_similarity']:.9f}")
    return 0


# --- command line ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    server = base.server_options("-m <embedding dir>")

    replay = modes.add_parser(
        "replay", parents=[server], help="send the requests one at a time"
    )
    replay.set_defaults(run=run_replay)

    concurrent = modes.add_parser(
        "concurrent", parents=[server], help="send the requests with N in flight"
    )
    concurrent.add_argument(
        "--in-flight",
        type=base.positive_int,
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
        "--starts", type=base.positive_int, required=True, help="number of fresh server starts"
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
        type=base.positive_seconds,
        required=True,
        metavar="SECONDS",
        help=(
            "how long a large request is left open before its connection is "
            "closed unread"
        ),
    )
    abandon.add_argument(
        "--min-texts",
        type=base.positive_int,
        required=True,
        help="a request with at least this many texts is a large request",
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
    except (base.InputError, base.ServerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
