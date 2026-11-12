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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{EmbeddingCommand, EmbeddingWork, EmbeddingWorker, EmbeddingWorkerProvider, Reply};
use crate::embeddings::stub::{STUB_DIM, STUB_MAX_LENGTH, STUB_VOCAB_SIZE, stub_loaded_model};
use crate::embeddings::{
    EmbedOptions, EmbedReply, EmbeddingBatch, EmbeddingEngine, EmbeddingModel, EmbeddingOutput,
    LoadedEmbeddingModel,
};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};

fn provider() -> EmbeddingWorkerProvider {
    EmbeddingWorkerProvider::from_loader(
        "stub-embed".to_string(),
        4,
        64,
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_model(false)),
    )
    .expect("stub worker spawns")
}

#[test]
fn worker_loads_stub_and_reports_info() {
    let p = provider();
    assert_eq!(p.model_id(), "stub-embed");
    assert_eq!(p.dim(), STUB_DIM);
    assert_eq!(p.max_length(), STUB_MAX_LENGTH);
    assert_eq!(p.vocab_size(), STUB_VOCAB_SIZE);
    assert!(!p.multi_vector());
    assert!(!p.supports_images());
    assert_eq!(p.info().batch_size, 4);
    assert_eq!(p.info().token_budget, Some(64));
    assert!(p.created_at() > 0);
}

#[test]
fn worker_turns_the_token_budget_off_at_zero() {
    let p = EmbeddingWorkerProvider::from_loader(
        "stub-embed".to_string(),
        4,
        0,
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_model(false)),
    )
    .expect("stub worker spawns");
    assert_eq!(p.info().token_budget, None);
}

#[test]
fn worker_round_trips_texts_and_tokens_on_its_own_thread() {
    let p = provider();
    let reply = p
        .embed_texts(
            vec!["hello world".to_string(), "hello".to_string()],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect("texts embed");
    assert_eq!(reply.vectors.len(), 2);
    // [CLS] hello world [SEP] + [CLS] hello [SEP]
    assert_eq!(reply.prompt_tokens, 7);
    let dot: f32 = reply.vectors[0]
        .values
        .iter()
        .zip(&reply.vectors[1].values)
        .map(|(a, b)| a * b)
        .sum();
    assert!(dot > 0.5, "texts sharing `hello` are similar: {dot}");

    let reply = p
        .embed_tokens(
            vec![vec![3, 4], vec![5]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect("tokens embed");
    assert_eq!(reply.prompt_tokens, 3);
    assert_eq!(reply.vectors[1].shape, vec![STUB_DIM]);
}

#[test]
fn worker_maps_engine_invalid_input() {
    let p = provider();
    let err = p
        .embed_tokens(
            vec![vec![STUB_VOCAB_SIZE as u32]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("out-of-vocab id is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
    let err = p
        .embed_texts(
            vec![String::new()],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("empty text is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
}

#[test]
fn worker_surfaces_loader_failure() {
    let result = EmbeddingWorkerProvider::from_loader(
        "broken".to_string(),
        4,
        4096,
        8,
        Duration::from_secs(30),
        || Err(anyhow::anyhow!("synthetic load failure")),
    );
    let err = result.expect_err("loader failure propagates");
    assert!(err.to_string().contains("synthetic load failure"), "{err}");
}

/// A model whose forward pass panics on a sentinel id, to prove the worker
/// survives and keeps serving.
struct PanickingModel;

impl EmbeddingModel for PanickingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
        let ids = mlxcel_core::utils::array_to_vec_f32(batch.input_ids);
        if ids.contains(&7.0) {
            panic!("synthetic embed panic");
        }
        let b = mlxcel_core::array_shape(batch.input_ids)[0];
        Ok(EmbeddingOutput {
            embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> crate::embeddings::PoolingMode {
        crate::embeddings::PoolingMode::Mean
    }

    fn embedding_dim(&self) -> usize {
        STUB_DIM
    }
}

fn panicking_loaded_model() -> LoadedEmbeddingModel {
    let mut loaded = stub_loaded_model(false);
    loaded.model = Box::new(PanickingModel);
    loaded
}

#[test]
fn worker_survives_a_panicking_request_and_keeps_serving() {
    let p = EmbeddingWorkerProvider::from_loader(
        "panicky".to_string(),
        4,
        4096,
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded_model()),
    )
    .expect("worker spawns");

    let err = p
        .embed_tokens(
            vec![vec![7]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("the sentinel id panics inside the engine");
    match err {
        EmbeddingError::Internal(message) => {
            assert!(message.contains("panic"), "{message}");
            assert!(!message.contains("no longer running"), "{message}");
        }
        other => panic!("expected Internal, got {other:?}"),
    }

    let reply = p
        .embed_tokens(
            vec![vec![3]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect("the worker still serves after the panic");
    assert_eq!(reply.vectors.len(), 1);
}

#[test]
fn request_timeout_returns_timeout_error() {
    struct SlowModel;
    impl EmbeddingModel for SlowModel {
        fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
            std::thread::sleep(Duration::from_millis(400));
            let b = mlxcel_core::array_shape(batch.input_ids)[0];
            Ok(EmbeddingOutput {
                embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
                last_hidden_state: None,
            })
        }
        fn default_pooling(&self) -> crate::embeddings::PoolingMode {
            crate::embeddings::PoolingMode::Mean
        }
        fn embedding_dim(&self) -> usize {
            STUB_DIM
        }
    }

    let p = EmbeddingWorkerProvider::from_loader(
        "slow".to_string(),
        4,
        4096,
        8,
        Duration::from_millis(50),
        || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(SlowModel);
            Ok(loaded)
        },
    )
    .expect("worker spawns");
    let err = p
        .embed_tokens(
            vec![vec![3]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("slower than the timeout");
    assert!(matches!(err, EmbeddingError::Timeout), "{err:?}");
}

#[test]
fn queue_full_returns_queuefull_error() {
    struct BlockingModel {
        gate: std::cell::Cell<Option<mpsc::Receiver<()>>>,
        started: mpsc::Sender<()>,
    }
    impl EmbeddingModel for BlockingModel {
        fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
            if let Some(gate) = self.gate.take() {
                let _ = self.started.send(());
                let _ = gate.recv();
            }
            let b = mlxcel_core::array_shape(batch.input_ids)[0];
            Ok(EmbeddingOutput {
                embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
                last_hidden_state: None,
            })
        }
        fn default_pooling(&self) -> crate::embeddings::PoolingMode {
            crate::embeddings::PoolingMode::Mean
        }
        fn embedding_dim(&self) -> usize {
            STUB_DIM
        }
    }

    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let queue_depth = 2;
    let worker = EmbeddingWorker::spawn(
        "embed-test-queuefull",
        queue_depth,
        Duration::from_millis(150),
        4,
        4096,
        move || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(BlockingModel {
                gate: std::cell::Cell::new(Some(gate_rx)),
                started: started_tx,
            });
            Ok(loaded)
        },
    )
    .expect("worker spawns");

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let _ = worker.embed_tokens(
                vec![vec![3]],
                EmbedOptions::default(),
                CancellationToken::new(),
            );
        });
        started_rx.recv().expect("in-flight request started");

        let mut saw_queue_full = false;
        for _ in 0..(queue_depth + 1) {
            match worker.embed_tokens(
                vec![vec![3]],
                EmbedOptions::default(),
                CancellationToken::new(),
            ) {
                Err(EmbeddingError::QueueFull) => {
                    saw_queue_full = true;
                    break;
                }
                Err(EmbeddingError::Timeout) => continue,
                other => panic!("unexpected result while filling the queue: {other:?}"),
            }
        }
        assert!(saw_queue_full, "a full bounded queue must reject admission");
        drop(gate_tx);
    });
}

/// A stub-backed worker whose cache release only counts its calls.
fn counting_worker<L>(loader: L) -> (EmbeddingWorker, Arc<AtomicUsize>)
where
    L: FnOnce() -> anyhow::Result<LoadedEmbeddingModel> + Send + 'static,
{
    let clears = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&clears);
    let worker = EmbeddingWorker::spawn_with_cache_clear(
        "embedding-worker-test",
        8,
        Duration::from_secs(30),
        4,
        4096,
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
    let (worker, clears) = counting_worker(|| Ok(stub_loaded_model(false)));
    // Dropping the worker joins its thread, so the count is final.
    drop(worker);
    assert_eq!(clears.load(Ordering::SeqCst), 0);
}

#[test]
fn each_request_releases_the_cache_once() {
    let (worker, clears) = counting_worker(|| Ok(stub_loaded_model(false)));

    // The text count does not change the count: the worker releases per
    // request, not per micro-batch (the batch size here is 4).
    let texts: Vec<String> = (0..9).map(|i| format!("text {i}")).collect();
    worker
        .embed_texts(texts, EmbedOptions::default(), CancellationToken::new())
        .expect("texts embed");
    assert_eq!(clears.load(Ordering::SeqCst), 1);

    worker
        .embed_tokens(
            vec![vec![3, 4], vec![5]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect("tokens embed");
    assert_eq!(clears.load(Ordering::SeqCst), 2);
}

#[test]
fn a_failed_request_still_releases_the_cache() {
    let (worker, clears) = counting_worker(|| Ok(stub_loaded_model(false)));
    worker
        .embed_tokens(
            vec![vec![STUB_VOCAB_SIZE as u32]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("out-of-vocab id is rejected");
    assert_eq!(clears.load(Ordering::SeqCst), 1);

    let image = crate::embeddings::ImageInput {
        image: image::DynamicImage::new_rgb8(4, 4),
    };
    worker
        .embed_image(image, EmbedOptions::default(), CancellationToken::new())
        .expect_err("the text-only stub rejects images");
    assert_eq!(clears.load(Ordering::SeqCst), 2);
}

#[test]
fn a_panicking_request_still_releases_the_cache() {
    let (worker, clears) = counting_worker(|| Ok(panicking_loaded_model()));
    worker
        .embed_tokens(
            vec![vec![7]],
            EmbedOptions::default(),
            CancellationToken::new(),
        )
        .expect_err("the sentinel id panics inside the engine");
    assert_eq!(clears.load(Ordering::SeqCst), 1);
}

/// Bounds every blocking receive of the gated tests, so a regression fails
/// instead of hanging.
const WAIT: Duration = Duration::from_secs(10);

/// A model that accepts images and counts its forward passes, so a test can
/// show that a request ran as several micro-batches.
struct ForwardCountingModel {
    forwards: Arc<AtomicUsize>,
}

impl EmbeddingModel for ForwardCountingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
        self.forwards.fetch_add(1, Ordering::SeqCst);
        let b = mlxcel_core::array_shape(batch.input_ids)[0];
        Ok(EmbeddingOutput {
            embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> crate::embeddings::PoolingMode {
        crate::embeddings::PoolingMode::Mean
    }

    fn embedding_dim(&self) -> usize {
        STUB_DIM
    }

    fn supports_images(&self) -> bool {
        true
    }
}

fn embed_command(work: EmbeddingWork, respond: Reply) -> EmbeddingCommand {
    EmbeddingCommand::Embed {
        work,
        opts: EmbedOptions::default(),
        cancel: CancellationToken::new(),
        respond,
    }
}

fn nine_texts(respond: Reply) -> EmbeddingCommand {
    embed_command(EmbeddingWork::Texts(vec!["hello".to_string(); 9]), respond)
}

fn two_token_rows(respond: Reply) -> EmbeddingCommand {
    embed_command(EmbeddingWork::Tokens(vec![vec![3, 4], vec![5]]), respond)
}

fn one_image(respond: Reply) -> EmbeddingCommand {
    embed_command(
        EmbeddingWork::Image(Box::new(crate::embeddings::ImageInput {
            image: image::DynamicImage::new_rgb8(4, 4),
        })),
        respond,
    )
}

#[test]
fn each_request_kind_releases_the_cache_once_before_its_reply() {
    let forwards = Arc::new(AtomicUsize::new(0));
    let model_forwards = Arc::clone(&forwards);
    let clears = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&clears);
    let (entered_tx, release_entered) = mpsc::channel::<()>();
    let (release_gate, gate_rx) = mpsc::channel::<()>();
    let worker = EmbeddingWorker::spawn_with_cache_clear(
        "embedding-worker-test-gated",
        8,
        Duration::from_secs(30),
        4,
        4096,
        move || {
            let _ = entered_tx.send(());
            let _ = gate_rx.recv();
            counter.fetch_add(1, Ordering::SeqCst);
        },
        move || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(ForwardCountingModel {
                forwards: model_forwards,
            });
            Ok(loaded)
        },
    )
    .expect("worker spawns");
    // Rebound after the worker so it drops first: a failed assertion then
    // opens the gate before the worker joins its thread.
    let release_gate = release_gate;

    // Nine texts at batch size 4 are three micro-batches.
    let requests: [(&str, fn(Reply) -> EmbeddingCommand, usize); 3] = [
        ("texts", nine_texts, 3),
        ("tokens", two_token_rows, 1),
        ("image", one_image, 1),
    ];
    let mut expected_forwards = 0;
    for (served, (kind, command, request_forwards)) in requests.into_iter().enumerate() {
        let (respond, reply) = mpsc::channel();
        worker
            .dispatch(command(respond))
            .expect("the worker accepts the request");
        release_entered
            .recv_timeout(WAIT)
            .expect("the worker releases the cache");
        // The worker is held inside the release, so a reply sent before the
        // release is already waiting here.
        assert!(
            matches!(reply.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "{kind}: the reply was sent before the cache release"
        );
        release_gate
            .send(())
            .expect("the release waits at the gate");
        reply
            .recv_timeout(WAIT)
            .expect("the worker replies")
            .unwrap_or_else(|err| panic!("{kind}: request failed: {err:?}"));
        expected_forwards += request_forwards;
        assert_eq!(forwards.load(Ordering::SeqCst), expected_forwards, "{kind}");
        assert_eq!(clears.load(Ordering::SeqCst), served + 1, "{kind}");
    }

    drop(release_gate);
    // Dropping the worker joins its thread, so the count is final.
    drop(worker);
    assert_eq!(clears.load(Ordering::SeqCst), 3);
}

#[test]
fn request_that_is_not_abandoned_returns_the_engine_result() {
    // Micro-batches of 2 over 5 rows of different lengths: 3 forward passes,
    // run in length order and written back in the caller's order.
    let p = EmbeddingWorkerProvider::from_loader(
        "stub-embed".to_string(),
        2,
        crate::embeddings::DEFAULT_EMBEDDING_TOKEN_BUDGET,
        8,
        Duration::from_secs(30),
        || Ok(stub_loaded_model(false)),
    )
    .expect("stub worker spawns");
    let engine = EmbeddingEngine::new(stub_loaded_model(false), 2);
    let opts = EmbedOptions::default();

    let rows = vec![
        vec![3, 4, 5],
        vec![6],
        vec![7, 8],
        vec![3],
        vec![9, 10, 11, 12],
    ];
    let expected = engine.embed_tokens(&rows, &opts).expect("engine embeds");
    let reply = p
        .embed_tokens(rows, opts.clone(), CancellationToken::new())
        .expect("worker embeds");
    assert_eq!(reply, expected);

    let texts: Vec<String> = ["hello world", "world", "hello", "world hello world"]
        .iter()
        .map(|t| t.to_string())
        .collect();
    let expected = engine.embed_texts(&texts, &opts).expect("engine embeds");
    let reply = p
        .embed_texts(texts, opts, CancellationToken::new())
        .expect("worker embeds");
    assert_eq!(reply, expected);
}

const GATED_BATCH_SIZE: usize = 2;

/// A fake model that counts the micro-batches it runs and holds each one at a
/// gate: it reports the batch's running index, then waits until the test
/// releases the batch or drops the release sender.
struct GatedBatchModel {
    batches_run: Arc<AtomicUsize>,
    batch_entered: mpsc::Sender<usize>,
    batch_release: mpsc::Receiver<()>,
}

impl EmbeddingModel for GatedBatchModel {
    fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
        let index = self.batches_run.fetch_add(1, Ordering::SeqCst);
        let _ = self.batch_entered.send(index);
        let _ = self.batch_release.recv_timeout(WAIT);
        let b = mlxcel_core::array_shape(batch.input_ids)[0];
        Ok(EmbeddingOutput {
            embeddings: mlxcel_core::ones(&[b, STUB_DIM as i32], mlxcel_core::dtype::FLOAT32),
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> crate::embeddings::PoolingMode {
        crate::embeddings::PoolingMode::Mean
    }

    fn embedding_dim(&self) -> usize {
        STUB_DIM
    }
}

/// A worker over [`GatedBatchModel`] with the test's ends of its channels.
/// `batch_release` is declared first so it drops before the worker joins its
/// thread, which may be waiting at the gate.
struct GatedWorker {
    batch_release: mpsc::Sender<()>,
    batch_entered: mpsc::Receiver<usize>,
    worker: EmbeddingWorker,
    batches_run: Arc<AtomicUsize>,
    clears: Arc<AtomicUsize>,
}

fn gated_worker(request_timeout: Duration) -> GatedWorker {
    let (entered_tx, batch_entered) = mpsc::channel();
    let (batch_release, release_rx) = mpsc::channel();
    let batches_run = Arc::new(AtomicUsize::new(0));
    let batches = Arc::clone(&batches_run);
    let clears = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&clears);
    let worker = EmbeddingWorker::spawn_with_cache_clear(
        "embed-test-gated",
        8,
        request_timeout,
        GATED_BATCH_SIZE,
        crate::embeddings::DEFAULT_EMBEDDING_TOKEN_BUDGET,
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
        },
        move || {
            let mut loaded = stub_loaded_model(false);
            loaded.model = Box::new(GatedBatchModel {
                batches_run: batches,
                batch_entered: entered_tx,
                batch_release: release_rx,
            });
            Ok(loaded)
        },
    )
    .expect("worker spawns");
    GatedWorker {
        batch_release,
        batch_entered,
        worker,
        batches_run,
        clears,
    }
}

/// `count` single-token rows, so `count / GATED_BATCH_SIZE` micro-batches.
fn token_rows(count: u32) -> Vec<Vec<u32>> {
    (0..count).map(|i| vec![3 + i]).collect()
}

#[test]
fn queued_request_whose_caller_is_gone_runs_zero_micro_batches() {
    let gated = gated_worker(Duration::from_secs(30));
    let worker = &gated.worker;
    let release = gated.batch_release;

    std::thread::scope(|scope| {
        // The first request holds the worker at the gate of its only batch.
        let first = scope.spawn(|| {
            worker.embed_tokens(
                token_rows(1),
                EmbedOptions::default(),
                CancellationToken::new(),
            )
        });
        assert_eq!(gated.batch_entered.recv_timeout(WAIT), Ok(0));

        // The second request (2 micro-batches) is cancelled before the worker
        // can dequeue it. It is enqueued either behind the held first request
        // or after it; the worker reaches it cancelled in both orders.
        let cancel = CancellationToken::new();
        cancel.cancel();
        let second =
            scope.spawn(|| worker.embed_tokens(token_rows(4), EmbedOptions::default(), cancel));

        release.send(()).expect("the worker waits at the gate");
        first
            .join()
            .expect("first caller joins")
            .expect("the request whose caller waits is run");
        // A wrongly run second request must not wait at the gate.
        drop(release);
        let err = second
            .join()
            .expect("second caller joins")
            .expect_err("the cancelled request is skipped");
        assert!(matches!(err, EmbeddingError::Cancelled), "{err:?}");
    });

    assert_eq!(
        gated.batches_run.load(Ordering::SeqCst),
        1,
        "only the first request's micro-batch runs"
    );
    // The skipped request allocated no MLX buffers, so it releases no cache.
    assert_eq!(gated.clears.load(Ordering::SeqCst), 1);
}

/// A request of `count` items of one input kind, each item one micro-batch
/// slot, so `count / GATED_BATCH_SIZE` micro-batches.
type GatedRequest =
    fn(&EmbeddingWorker, u32, CancellationToken) -> Result<EmbedReply, EmbeddingError>;

fn token_request(
    worker: &EmbeddingWorker,
    count: u32,
    cancel: CancellationToken,
) -> Result<EmbedReply, EmbeddingError> {
    worker.embed_tokens(token_rows(count), EmbedOptions::default(), cancel)
}

fn text_request(
    worker: &EmbeddingWorker,
    count: u32,
    cancel: CancellationToken,
) -> Result<EmbedReply, EmbeddingError> {
    let texts = vec!["hello".to_string(); count as usize];
    worker.embed_texts(texts, EmbedOptions::default(), cancel)
}

#[test]
fn in_flight_request_stops_at_next_micro_batch_boundary_and_serves_next() {
    in_flight_request_stops_and_the_next_is_served(token_request);
}

#[test]
fn in_flight_text_request_stops_at_next_micro_batch_boundary_and_serves_next() {
    in_flight_request_stops_and_the_next_is_served(text_request);
}

fn in_flight_request_stops_and_the_next_is_served(request: GatedRequest) {
    let gated = gated_worker(Duration::from_secs(30));
    let worker = &gated.worker;
    let release = gated.batch_release;
    let cancel = CancellationToken::new();

    std::thread::scope(|scope| {
        // 6 items are 3 micro-batches.
        let caller_cancel = cancel.clone();
        let stopped = scope.spawn(|| request(worker, 6, caller_cancel));
        assert_eq!(gated.batch_entered.recv_timeout(WAIT), Ok(0));

        // Batch 0 is held at the gate, so the cancellation lands inside it;
        // the release then lets batch 0 finish and reach the boundary check.
        cancel.cancel();
        release.send(()).expect("the worker waits at the gate");
        let err = stopped
            .join()
            .expect("caller joins")
            .expect_err("the request stops before batch 1");
        assert!(matches!(err, EmbeddingError::Cancelled), "{err:?}");
    });
    assert_eq!(
        gated.batches_run.load(Ordering::SeqCst),
        1,
        "batches 1 and 2 of the stopped request do not run"
    );
    // The stopped request ran a micro-batch, so it releases the cache once.
    assert_eq!(gated.clears.load(Ordering::SeqCst), 1);

    // With the release sender gone the gate holds no batch, so the next
    // request (2 micro-batches) runs straight through.
    drop(release);
    let live = request(worker, 4, CancellationToken::new())
        .expect("the worker serves the next request after a stopped one");
    assert_eq!(live.vectors.len(), 4);
    assert_eq!(gated.batches_run.load(Ordering::SeqCst), 3);
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
        .embed_tokens(token_rows(1), EmbedOptions::default(), cancel.clone())
        .expect_err("no reply while the gate holds the batch");
    assert!(matches!(err, EmbeddingError::Timeout), "{err:?}");
    assert!(cancel.is_cancelled());
}
