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

use super::{EmbeddingCommand, EmbeddingWorker, EmbeddingWorkerProvider, Reply};
use crate::embeddings::stub::{STUB_DIM, STUB_MAX_LENGTH, STUB_VOCAB_SIZE, stub_loaded_model};
use crate::embeddings::{
    EmbedOptions, EmbeddingBatch, EmbeddingModel, EmbeddingOutput, LoadedEmbeddingModel,
};
use crate::server::embedding_model::{EmbeddingError, EmbeddingModelProvider};

fn provider() -> EmbeddingWorkerProvider {
    EmbeddingWorkerProvider::from_loader(
        "stub-embed".to_string(),
        4,
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
    assert!(p.created_at() > 0);
}

#[test]
fn worker_round_trips_texts_and_tokens_on_its_own_thread() {
    let p = provider();
    let reply = p
        .embed_texts(
            vec!["hello world".to_string(), "hello".to_string()],
            EmbedOptions::default(),
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
        .embed_tokens(vec![vec![3, 4], vec![5]], EmbedOptions::default())
        .expect("tokens embed");
    assert_eq!(reply.prompt_tokens, 3);
    assert_eq!(reply.vectors[1].shape, vec![STUB_DIM]);
}

#[test]
fn worker_maps_engine_invalid_input() {
    let p = provider();
    let err = p
        .embed_tokens(vec![vec![STUB_VOCAB_SIZE as u32]], EmbedOptions::default())
        .expect_err("out-of-vocab id is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
    let err = p
        .embed_texts(vec![String::new()], EmbedOptions::default())
        .expect_err("empty text is rejected");
    assert!(matches!(err, EmbeddingError::InvalidInput(_)), "{err:?}");
}

#[test]
fn worker_surfaces_loader_failure() {
    let result = EmbeddingWorkerProvider::from_loader(
        "broken".to_string(),
        4,
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
        8,
        Duration::from_secs(30),
        || Ok(panicking_loaded_model()),
    )
    .expect("worker spawns");

    let err = p
        .embed_tokens(vec![vec![7]], EmbedOptions::default())
        .expect_err("the sentinel id panics inside the engine");
    match err {
        EmbeddingError::Internal(message) => {
            assert!(message.contains("panic"), "{message}");
            assert!(!message.contains("no longer running"), "{message}");
        }
        other => panic!("expected Internal, got {other:?}"),
    }

    let reply = p
        .embed_tokens(vec![vec![3]], EmbedOptions::default())
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
        .embed_tokens(vec![vec![3]], EmbedOptions::default())
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
            let _ = worker.embed_tokens(vec![vec![3]], EmbedOptions::default());
        });
        started_rx.recv().expect("in-flight request started");

        let mut saw_queue_full = false;
        for _ in 0..(queue_depth + 1) {
            match worker.embed_tokens(vec![vec![3]], EmbedOptions::default()) {
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
        .embed_texts(texts, EmbedOptions::default())
        .expect("texts embed");
    assert_eq!(clears.load(Ordering::SeqCst), 1);

    worker
        .embed_tokens(vec![vec![3, 4], vec![5]], EmbedOptions::default())
        .expect("tokens embed");
    assert_eq!(clears.load(Ordering::SeqCst), 2);
}

#[test]
fn a_failed_request_still_releases_the_cache() {
    let (worker, clears) = counting_worker(|| Ok(stub_loaded_model(false)));
    worker
        .embed_tokens(vec![vec![STUB_VOCAB_SIZE as u32]], EmbedOptions::default())
        .expect_err("out-of-vocab id is rejected");
    assert_eq!(clears.load(Ordering::SeqCst), 1);

    let image = crate::embeddings::ImageInput {
        image: image::DynamicImage::new_rgb8(4, 4),
    };
    worker
        .embed_image(image, EmbedOptions::default())
        .expect_err("the text-only stub rejects images");
    assert_eq!(clears.load(Ordering::SeqCst), 2);
}

#[test]
fn a_panicking_request_still_releases_the_cache() {
    let (worker, clears) = counting_worker(|| Ok(panicking_loaded_model()));
    worker
        .embed_tokens(vec![vec![7]], EmbedOptions::default())
        .expect_err("the sentinel id panics inside the engine");
    assert_eq!(clears.load(Ordering::SeqCst), 1);
}

/// Bounds every blocking receive of the gated test, so a regression fails
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

fn nine_texts(respond: Reply) -> EmbeddingCommand {
    EmbeddingCommand::Texts {
        texts: vec!["hello".to_string(); 9],
        opts: EmbedOptions::default(),
        respond,
    }
}

fn two_token_rows(respond: Reply) -> EmbeddingCommand {
    EmbeddingCommand::Tokens {
        rows: vec![vec![3, 4], vec![5]],
        opts: EmbedOptions::default(),
        respond,
    }
}

fn one_image(respond: Reply) -> EmbeddingCommand {
    EmbeddingCommand::Image {
        image: Box::new(crate::embeddings::ImageInput {
            image: image::DynamicImage::new_rgb8(4, 4),
        }),
        opts: EmbedOptions::default(),
        respond,
    }
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
