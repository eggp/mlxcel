// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rerank worker admission, timeout, panic-recovery and cache-release gates, over the stub
//! reranker so no checkpoint is needed.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{RerankWorker, RerankWorkerProvider};
use crate::rerank::stub::{STUB_BATCH_SIZE, STUB_MAX_LENGTH, stub_loaded_reranker};
use crate::rerank::{
    LoadedReranker, RerankAbandoned, RerankItem, RerankScores, Reranker, RerankerKind,
};
use crate::server::rerank_model::{RerankError, RerankModelProvider};

fn provider() -> RerankWorkerProvider {
    RerankWorkerProvider::from_loader(
        "stub-rerank".to_string(),
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_reranker(RerankerKind::GenerativeText, false)),
    )
    .expect("stub worker spawns")
}

fn query() -> RerankItem {
    RerankItem::text("alpha beta")
}

#[test]
fn worker_loads_stub_and_reports_info() {
    let p = provider();
    assert_eq!(p.model_id(), "stub-rerank");
    assert_eq!(p.kind(), RerankerKind::GenerativeText);
    assert_eq!(p.max_length(), STUB_MAX_LENGTH);
    assert!(!p.supports_images());
    assert_eq!(p.info().batch_size, STUB_BATCH_SIZE);
    assert_eq!(p.info().model_type, "stub");
    assert!(p.created_at() > 0);
}

#[test]
fn worker_round_trips_scores_on_its_own_thread() {
    let p = provider();
    let scored = p
        .rerank(
            query(),
            vec![RerankItem::text("alpha beta"), RerankItem::text("gamma")],
            None,
            CancellationToken::new(),
        )
        .expect("scoring succeeds");
    assert_eq!(scored.scores.len(), 2);
    assert!(scored.scores[0] > scored.scores[1]);
    assert_eq!(scored.prompt_tokens, 4 + 3);
}

#[test]
fn worker_maps_a_family_error_to_internal() {
    let p = provider();
    let image = crate::rerank::ImageInput {
        image: image::DynamicImage::new_rgb8(4, 4),
    };
    let err = p
        .rerank(
            query(),
            vec![RerankItem::image(image)],
            None,
            CancellationToken::new(),
        )
        .expect_err("the text-only stub rejects images");
    match err {
        RerankError::Internal(message) => {
            assert!(message.contains("does not accept images"), "{message}")
        }
        other => panic!("expected Internal, got {other:?}"),
    }
}

#[test]
fn worker_surfaces_loader_failure() {
    let result =
        RerankWorkerProvider::from_loader("broken".to_string(), 8, Duration::from_secs(30), || {
            Err(anyhow::anyhow!("synthetic load failure"))
        });
    let err = result.expect_err("loader failure propagates");
    assert!(err.to_string().contains("synthetic load failure"), "{err}");
}

/// A reranker whose scoring panics on a sentinel document, to prove the
/// worker survives and keeps serving.
struct PanickingReranker;

impl Reranker for PanickingReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score_until(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
        _is_abandoned: &dyn Fn() -> bool,
    ) -> anyhow::Result<RerankScores> {
        if documents.iter().any(|d| d.text_or_empty() == "boom") {
            panic!("synthetic rerank panic");
        }
        Ok(RerankScores {
            scores: vec![0.5; documents.len()],
            prompt_tokens: documents.len(),
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

fn panicking_loaded() -> LoadedReranker {
    LoadedReranker {
        reranker: Box::new(PanickingReranker),
        kind: RerankerKind::GenerativeText,
        model_type: "panicky".to_string(),
    }
}

#[test]
fn worker_survives_a_panicking_request_and_keeps_serving() {
    let p = RerankWorkerProvider::from_loader(
        "panicky".to_string(),
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded()),
    )
    .expect("worker spawns");

    let err = p
        .rerank(
            query(),
            vec![RerankItem::text("boom")],
            None,
            CancellationToken::new(),
        )
        .expect_err("the sentinel document panics inside the family");
    match err {
        RerankError::Internal(message) => {
            assert!(message.contains("panic"), "{message}");
            assert!(!message.contains("no longer running"), "{message}");
        }
        other => panic!("expected Internal, got {other:?}"),
    }

    let scored = p
        .rerank(
            query(),
            vec![RerankItem::text("fine")],
            None,
            CancellationToken::new(),
        )
        .expect("the worker still serves after the panic");
    assert_eq!(scored.scores.len(), 1);
}

/// A reranker that sleeps, for the timeout gate.
struct SlowReranker(Duration);

impl Reranker for SlowReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score_until(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
        _is_abandoned: &dyn Fn() -> bool,
    ) -> anyhow::Result<RerankScores> {
        std::thread::sleep(self.0);
        Ok(RerankScores {
            scores: vec![0.25; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

#[test]
fn request_timeout_returns_timeout_error() {
    let p =
        RerankWorkerProvider::from_loader("slow".to_string(), 8, Duration::from_millis(50), || {
            Ok(LoadedReranker {
                reranker: Box::new(SlowReranker(Duration::from_millis(400))),
                kind: RerankerKind::GenerativeText,
                model_type: "slow".to_string(),
            })
        })
        .expect("worker spawns");
    let err = p
        .rerank(
            query(),
            vec![RerankItem::text("alpha")],
            None,
            CancellationToken::new(),
        )
        .expect_err("slower than the timeout");
    assert!(matches!(err, RerankError::Timeout), "{err:?}");
}

/// A reranker that blocks on a gate until the test releases it, so the
/// bounded queue can be filled deterministically.
struct BlockingReranker {
    gate: std::cell::Cell<Option<mpsc::Receiver<()>>>,
    started: mpsc::Sender<()>,
}

impl Reranker for BlockingReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score_until(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
        _is_abandoned: &dyn Fn() -> bool,
    ) -> anyhow::Result<RerankScores> {
        if let Some(gate) = self.gate.take() {
            let _ = self.started.send(());
            let _ = gate.recv();
        }
        Ok(RerankScores {
            scores: vec![0.5; documents.len()],
            prompt_tokens: 1,
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        1
    }
}

#[test]
fn queue_full_returns_queuefull_error() {
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let queue_depth = 2;
    // A short reply timeout keeps the queue-filling probes from blocking on a
    // worker that is deliberately parked; each one times out and the next
    // takes its slot until admission itself is refused.
    let worker = super::RerankWorker::spawn(
        "rerank-test-queuefull",
        queue_depth,
        Duration::from_millis(150),
        move || {
            Ok(LoadedReranker {
                reranker: Box::new(BlockingReranker {
                    gate: std::cell::Cell::new(Some(gate_rx)),
                    started: started_tx,
                }),
                kind: RerankerKind::GenerativeText,
                model_type: "blocking".to_string(),
            })
        },
    )
    .expect("worker spawns");

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let _ = worker.score(
                query(),
                vec![RerankItem::text("a")],
                None,
                CancellationToken::new(),
            );
        });
        started_rx.recv().expect("in-flight request started");

        let mut saw_queue_full = false;
        for _ in 0..(queue_depth + 1) {
            match worker.score(
                query(),
                vec![RerankItem::text("b")],
                None,
                CancellationToken::new(),
            ) {
                Err(RerankError::QueueFull) => {
                    saw_queue_full = true;
                    break;
                }
                Err(RerankError::Timeout) => continue,
                other => panic!("unexpected result while filling the queue: {other:?}"),
            }
        }
        assert!(saw_queue_full, "a full bounded queue must reject admission");
        drop(gate_tx);
    });
}

/// A stub-backed worker whose cache release only counts its calls.
fn counting_worker<L>(loader: L) -> (RerankWorker, Arc<AtomicUsize>)
where
    L: FnOnce() -> anyhow::Result<LoadedReranker> + Send + 'static,
{
    let clears = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&clears);
    let worker = RerankWorker::spawn_with_cache_clear(
        "rerank-worker-test",
        8,
        Duration::from_secs(30),
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
        },
        loader,
    )
    .expect("worker spawns");
    (worker, clears)
}

#[test]
fn loading_and_shutting_down_release_no_cache() {
    let (worker, clears) =
        counting_worker(|| Ok(stub_loaded_reranker(RerankerKind::GenerativeText, false)));
    // Dropping the worker joins its thread, so the count is final.
    drop(worker);
    assert_eq!(clears.load(Ordering::SeqCst), 0);
}

#[test]
fn each_scoring_request_releases_the_cache_once() {
    let (worker, clears) =
        counting_worker(|| Ok(stub_loaded_reranker(RerankerKind::GenerativeText, false)));

    // The document count does not change the count: the worker releases
    // per request. The stub does not micro-batch, so this does not observe a
    // release inside a family's batch loop.
    let documents: Vec<RerankItem> = (0..STUB_BATCH_SIZE * 3 + 1)
        .map(|i| RerankItem::text(format!("doc {i}")))
        .collect();
    worker
        .score(query(), documents, None, CancellationToken::new())
        .expect("scoring succeeds");
    assert_eq!(clears.load(Ordering::SeqCst), 1);

    for _ in 0..3 {
        worker
            .score(
                query(),
                vec![RerankItem::text("alpha")],
                None,
                CancellationToken::new(),
            )
            .expect("scoring succeeds");
    }
    assert_eq!(clears.load(Ordering::SeqCst), 4);
}

#[test]
fn a_failed_scoring_request_still_releases_the_cache() {
    let (worker, clears) =
        counting_worker(|| Ok(stub_loaded_reranker(RerankerKind::GenerativeText, false)));
    let image = crate::rerank::ImageInput {
        image: image::DynamicImage::new_rgb8(4, 4),
    };
    worker
        .score(
            query(),
            vec![RerankItem::image(image)],
            None,
            CancellationToken::new(),
        )
        .expect_err("the text-only stub rejects images");
    assert_eq!(clears.load(Ordering::SeqCst), 1);
}

#[test]
fn a_panicking_scoring_request_still_releases_the_cache() {
    let (worker, clears) = counting_worker(|| Ok(panicking_loaded()));
    worker
        .score(
            query(),
            vec![RerankItem::text("boom")],
            None,
            CancellationToken::new(),
        )
        .expect_err("the sentinel document panics inside the family");
    assert_eq!(clears.load(Ordering::SeqCst), 1);
}

/// Bounds every blocking receive of the cancellation tests, so a regression
/// fails instead of hanging.
const WAIT: Duration = Duration::from_secs(10);

const GATED_BATCH_SIZE: usize = 2;

/// A fake reranker that counts the micro-batches it scores and holds each one
/// at a gate: it reports the batch index, then waits until the test releases
/// the batch or drops the release sender.
struct GatedBatchReranker {
    batches_scored: Arc<AtomicUsize>,
    batch_entered: mpsc::Sender<usize>,
    batch_release: mpsc::Receiver<()>,
}

impl Reranker for GatedBatchReranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score_until(
        &self,
        _query: &RerankItem,
        documents: &[RerankItem],
        _instruction: Option<&str>,
        is_abandoned: &dyn Fn() -> bool,
    ) -> anyhow::Result<RerankScores> {
        let mut scores = Vec::with_capacity(documents.len());
        for (batch_idx, chunk) in documents.chunks(GATED_BATCH_SIZE).enumerate() {
            if is_abandoned() {
                anyhow::bail!(RerankAbandoned);
            }
            self.batches_scored.fetch_add(1, Ordering::SeqCst);
            let _ = self.batch_entered.send(batch_idx);
            let _ = self.batch_release.recv_timeout(WAIT);
            for _ in chunk {
                scores.push(1.0 / (scores.len() + 1) as f32);
            }
        }
        Ok(RerankScores {
            scores,
            prompt_tokens: documents.len(),
        })
    }

    fn max_length(&self) -> usize {
        STUB_MAX_LENGTH
    }

    fn batch_size(&self) -> usize {
        GATED_BATCH_SIZE
    }
}

/// A worker over [`GatedBatchReranker`] with the test's ends of its channels.
/// `batch_release` is declared first so it drops before the worker joins its
/// thread, which may be waiting at the gate.
struct GatedWorker {
    batch_release: mpsc::Sender<()>,
    batch_entered: mpsc::Receiver<usize>,
    worker: RerankWorker,
    batches_scored: Arc<AtomicUsize>,
    clears: Arc<AtomicUsize>,
}

fn gated_worker(request_timeout: Duration) -> GatedWorker {
    let (entered_tx, batch_entered) = mpsc::channel();
    let (batch_release, release_rx) = mpsc::channel();
    let batches_scored = Arc::new(AtomicUsize::new(0));
    let batches = Arc::clone(&batches_scored);
    let clears = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&clears);
    let worker = RerankWorker::spawn_with_cache_clear(
        "rerank-test-gated",
        8,
        request_timeout,
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
        },
        move || {
            Ok(LoadedReranker {
                reranker: Box::new(GatedBatchReranker {
                    batches_scored: batches,
                    batch_entered: entered_tx,
                    batch_release: release_rx,
                }),
                kind: RerankerKind::GenerativeText,
                model_type: "gated".to_string(),
            })
        },
    )
    .expect("worker spawns");
    GatedWorker {
        batch_release,
        batch_entered,
        worker,
        batches_scored,
        clears,
    }
}

fn numbered_documents(count: usize) -> Vec<RerankItem> {
    (0..count)
        .map(|i| RerankItem::text(format!("doc {i}")))
        .collect()
}

#[test]
fn queued_request_whose_caller_is_gone_scores_zero_batches() {
    let gated = gated_worker(Duration::from_secs(30));
    let worker = &gated.worker;
    let release = gated.batch_release;

    std::thread::scope(|scope| {
        // The first request holds the worker at the gate of its only batch.
        let first = scope.spawn(|| {
            worker.score(
                query(),
                numbered_documents(1),
                None,
                CancellationToken::new(),
            )
        });
        assert_eq!(gated.batch_entered.recv_timeout(WAIT), Ok(0));

        // The second request (2 micro-batches) is cancelled before the worker
        // can dequeue it. It is enqueued either behind the held first request
        // or after it; the worker reaches it cancelled in both orders.
        let cancel = CancellationToken::new();
        cancel.cancel();
        let second = scope.spawn(|| worker.score(query(), numbered_documents(4), None, cancel));

        release.send(()).expect("the worker waits at the gate");
        first
            .join()
            .expect("first caller joins")
            .expect("the request whose caller waits is scored");
        // A wrongly scored second request must not wait at the gate.
        drop(release);
        let err = second
            .join()
            .expect("second caller joins")
            .expect_err("the cancelled request is skipped");
        assert!(matches!(err, RerankError::Cancelled), "{err:?}");
    });

    assert_eq!(
        gated.batches_scored.load(Ordering::SeqCst),
        1,
        "only the first request's batch is scored"
    );
    // The skipped request allocated no MLX buffers, so it releases no cache.
    assert_eq!(gated.clears.load(Ordering::SeqCst), 1);
}

#[test]
fn in_flight_request_stops_at_next_micro_batch_boundary_and_serves_next() {
    let gated = gated_worker(Duration::from_secs(30));
    let worker = &gated.worker;
    let release = gated.batch_release;
    let cancel = CancellationToken::new();

    std::thread::scope(|scope| {
        // 6 documents are 3 micro-batches.
        let caller_cancel = cancel.clone();
        let stopped =
            scope.spawn(|| worker.score(query(), numbered_documents(6), None, caller_cancel));
        assert_eq!(gated.batch_entered.recv_timeout(WAIT), Ok(0));

        // Batch 0 is held at the gate, so the cancellation lands inside it;
        // the release then lets batch 0 finish and reach the boundary check.
        cancel.cancel();
        release.send(()).expect("the worker waits at the gate");
        let err = stopped
            .join()
            .expect("caller joins")
            .expect_err("the request stops before batch 1");
        assert!(matches!(err, RerankError::Cancelled), "{err:?}");
    });
    assert_eq!(
        gated.batches_scored.load(Ordering::SeqCst),
        1,
        "batches 1 and 2 of the stopped request are not scored"
    );
    assert_eq!(gated.clears.load(Ordering::SeqCst), 1);

    // With the release sender gone the gate holds no batch, so the next
    // request (2 micro-batches) runs straight through.
    drop(release);
    let live = worker
        .score(
            query(),
            numbered_documents(4),
            None,
            CancellationToken::new(),
        )
        .expect("the worker serves the next request after a stopped one");
    assert_eq!(live.scores, vec![1.0, 0.5, 1.0 / 3.0, 0.25]);
    assert_eq!(gated.batches_scored.load(Ordering::SeqCst), 3);
    assert_eq!(gated.clears.load(Ordering::SeqCst), 2);
}

#[test]
fn reply_timeout_cancels_the_request_token() {
    let gated = gated_worker(Duration::from_millis(50));
    let cancel = CancellationToken::new();

    // The gate is not released while the caller waits, so no reply can arrive
    // and the timeout is the only outcome, however late the worker dequeues
    // the request.
    let err = gated
        .worker
        .score(query(), numbered_documents(1), None, cancel.clone())
        .expect_err("no reply while the gate holds the batch");
    assert!(matches!(err, RerankError::Timeout), "{err:?}");
    assert!(cancel.is_cancelled());
}
