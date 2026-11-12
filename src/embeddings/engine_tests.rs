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

use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

use super::engine::{
    DEFAULT_EMBEDDING_TOKEN_BUDGET, EmbedOptions, EmbeddingEngine, EmbeddingEngineError,
    micro_batch_bounds,
};
use super::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use super::pooling::PoolingMode;
use super::stub::{
    STUB_DIM, STUB_MAX_LENGTH, STUB_VOCAB_SIZE, StubEmbeddingModel, stub_loaded_model,
};

fn engine(batch_size: usize) -> EmbeddingEngine {
    EmbeddingEngine::new(stub_loaded_model(false), batch_size)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn engine_reports_stub_limits() {
    let e = engine(4);
    assert_eq!(e.dim(), STUB_DIM);
    assert_eq!(e.max_length(), STUB_MAX_LENGTH);
    assert_eq!(e.vocab_size(), STUB_VOCAB_SIZE);
    assert!(!e.multi_vector());
    assert!(!e.supports_images());
    assert_eq!(e.batch_size(), 4);
    assert_eq!(
        EmbeddingEngine::new(stub_loaded_model(false), 0).batch_size(),
        1
    );
}

#[test]
fn embed_tokens_returns_unit_vectors_in_request_order_across_micro_batches() {
    // batch_size 1 forces one micro-batch per row; the rows have different
    // lengths so the length sort reorders them and the write-back must undo it.
    let e = engine(1);
    let rows = vec![vec![2, 3, 4], vec![5], vec![2, 3]];
    let reply = e.embed_tokens(&rows, &EmbedOptions::default()).unwrap();
    assert_eq!(reply.prompt_tokens, 6);
    assert_eq!(reply.vectors.len(), 3);
    for (i, v) in reply.vectors.iter().enumerate() {
        assert_eq!(v.shape, vec![STUB_DIM], "row {i}");
        assert!((norm(&v.values) - 1.0).abs() < 1e-5, "row {i} is unit norm");
    }
    // Row 1 is one-hot on id 5; rows 0 and 2 share ids 2 and 3.
    assert!((reply.vectors[1].values[5] - 1.0).abs() < 1e-6);
    assert!(cosine(&reply.vectors[0].values, &reply.vectors[2].values) > 0.5);
    assert!(cosine(&reply.vectors[0].values, &reply.vectors[1].values).abs() < 1e-6);

    // A larger batch gives the same vectors: micro-batching is invisible.
    let wide = engine(16)
        .embed_tokens(&rows, &EmbedOptions::default())
        .unwrap();
    for (a, b) in reply.vectors.iter().zip(&wide.vectors) {
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}

#[test]
fn embed_texts_uses_the_tokenizer_and_counts_tokens() {
    let e = engine(8);
    let reply = e
        .embed_texts(&["hello".to_string()], &EmbedOptions::default())
        .unwrap();
    assert_eq!(reply.vectors.len(), 1);
    // [CLS] hello [SEP]: three real tokens, only `hello` (id 3) is one-hot.
    assert_eq!(reply.prompt_tokens, 3);
    assert!(
        (reply.vectors[0].values[3] - 1.0).abs() < 1e-6,
        "hello is id 3"
    );

    let empty = e.embed_texts(&[], &EmbedOptions::default()).unwrap();
    assert!(empty.vectors.is_empty());
    assert_eq!(empty.prompt_tokens, 0);
}

#[test]
fn embed_texts_rejects_empty_strings() {
    let e = engine(8);
    let err = e
        .embed_texts(
            &["hello".to_string(), String::new()],
            &EmbedOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(ref m) if m.contains("input[1]")));
}

#[test]
fn embed_tokens_rejects_out_of_vocab_ids() {
    let e = engine(8);
    let err = e
        .embed_tokens(&[vec![1, STUB_VOCAB_SIZE as u32]], &EmbedOptions::default())
        .unwrap_err();
    match err {
        EmbeddingEngineError::InvalidInput(m) => assert!(m.contains("vocab_size"), "{m}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    let err = e
        .embed_tokens(&[vec![]], &EmbedOptions::default())
        .unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(_)));
}

#[test]
fn dimensions_truncates_and_renormalizes() {
    let e = engine(8);
    let opts = EmbedOptions {
        instruction: None,
        dimensions: Some(4),
        normalize: None,
    };
    let reply = e.embed_tokens(&[vec![2, 3]], &opts).unwrap();
    let v = &reply.vectors[0];
    assert_eq!(v.shape, vec![4]);
    assert!(
        (norm(&v.values) - 1.0).abs() < 1e-5,
        "re-normalized after truncation"
    );
    // ids 2 and 3 both survive the cut to 4 dims and share the mass equally.
    assert!((v.values[2] - v.values[3]).abs() < 1e-6);

    for bad in [0, STUB_DIM + 1] {
        let err = e
            .embed_tokens(
                &[vec![2]],
                &EmbedOptions {
                    instruction: None,
                    dimensions: Some(bad),
                    normalize: None,
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, EmbeddingEngineError::InvalidInput(_)),
            "{bad}"
        );
    }
}

#[test]
fn multi_vector_reply_has_one_row_per_real_token() {
    let e = EmbeddingEngine::new(stub_loaded_model(true), 8);
    assert!(e.multi_vector());
    let reply = e
        .embed_tokens(&[vec![2, 3, 4], vec![5]], &EmbedOptions::default())
        .unwrap();
    assert_eq!(reply.vectors[0].shape, vec![3, STUB_DIM]);
    assert_eq!(reply.vectors[1].shape, vec![1, STUB_DIM]);
    let rows: Vec<&[f32]> = reply.vectors[0].rows().collect();
    assert_eq!(rows.len(), 3);
    assert!((rows[0][2] - 1.0).abs() < 1e-6);
    assert!((rows[2][4] - 1.0).abs() < 1e-6);
    assert!(reply.vectors[0].is_multi_vector());

    // Truncation applies per row.
    let reply = e
        .embed_tokens(
            &[vec![2, 3]],
            &EmbedOptions {
                instruction: None,
                dimensions: Some(3),
                normalize: None,
            },
        )
        .unwrap();
    assert_eq!(reply.vectors[0].shape, vec![2, 3]);
}

#[test]
fn engine_token_budget_defaults_to_4096_and_zero_turns_it_off() {
    assert_eq!(DEFAULT_EMBEDDING_TOKEN_BUDGET, 4096);
    assert_eq!(engine(4).token_budget(), Some(4096));
    assert_eq!(engine(4).with_token_budget(512).token_budget(), Some(512));
    assert_eq!(engine(4).with_token_budget(0).token_budget(), None);
}

/// The stub model, recording the `[B, L]` shape of every micro-batch it runs
/// and reporting `pad_to` as its fixed sequence width.
struct ShapeRecorder {
    inner: StubEmbeddingModel,
    pad_to: Option<usize>,
    shapes: Rc<RefCell<Vec<Vec<i32>>>>,
}

impl EmbeddingModel for ShapeRecorder {
    fn embed(&self, batch: &EmbeddingBatch) -> anyhow::Result<EmbeddingOutput> {
        self.shapes
            .borrow_mut()
            .push(mlxcel_core::array_shape(batch.input_ids));
        self.inner.embed(batch)
    }

    fn default_pooling(&self) -> PoolingMode {
        self.inner.default_pooling()
    }

    fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }

    fn pad_to_max_length(&self) -> Option<usize> {
        self.pad_to
    }
}

/// Micro-batch shapes for four one-token rows and two six-token rows,
/// interleaved, at `batch_size` 16 and the given budget.
fn recorded_micro_batches(token_budget: usize) -> Vec<Vec<i32>> {
    let shapes = Rc::new(RefCell::new(Vec::new()));
    let mut loaded = stub_loaded_model(false);
    loaded.model = Box::new(ShapeRecorder {
        inner: StubEmbeddingModel::new(),
        pad_to: None,
        shapes: Rc::clone(&shapes),
    });
    let e = EmbeddingEngine::new(loaded, 16).with_token_budget(token_budget);
    let rows = vec![vec![2; 6], vec![3], vec![4], vec![5; 6], vec![6], vec![7]];
    let reply = e.embed_tokens(&rows, &EmbedOptions::default()).unwrap();
    assert!((reply.vectors[0].values[2] - 1.0).abs() < 1e-6);
    assert!((reply.vectors[3].values[5] - 1.0).abs() < 1e-6);
    shapes.take()
}

#[test]
fn engine_cuts_micro_batches_by_the_token_budget_it_was_given() {
    // 5 rows x 6 tokens exceeds 12, so the long rows split off; 2 x 6 fits.
    assert_eq!(recorded_micro_batches(12), vec![vec![4, 1], vec![2, 6]]);
    assert_eq!(
        recorded_micro_batches(DEFAULT_EMBEDDING_TOKEN_BUDGET),
        vec![vec![6, 6]]
    );
    assert_eq!(recorded_micro_batches(0), vec![vec![6, 6]]);
}

#[test]
fn engine_counts_a_fixed_pad_width_against_the_token_budget() {
    let shapes = Rc::new(RefCell::new(Vec::new()));
    let mut loaded = stub_loaded_model(false);
    loaded.model = Box::new(ShapeRecorder {
        inner: StubEmbeddingModel::new(),
        pad_to: Some(8),
        shapes: Rc::clone(&shapes),
    });
    let e = EmbeddingEngine::new(loaded, 16).with_token_budget(16);
    let rows = vec![vec![3], vec![4], vec![5], vec![6]];
    e.embed_tokens(&rows, &EmbedOptions::default()).unwrap();
    // Every row runs 8 wide, so 2 x 8 fills the budget; counted at their raw
    // length of 1, all four rows would fit in one micro-batch.
    assert_eq!(shapes.take(), vec![vec![2, 8], vec![2, 8]]);
}

#[test]
fn micro_batch_bounds_without_budget_match_fixed_row_chunks() {
    let lengths: Vec<usize> = (1..=40).map(|n| n * 300).collect();
    for batch_size in [1, 3, 16, 40, 64] {
        let expected: Vec<Range<usize>> = (0..lengths.len())
            .step_by(batch_size)
            .map(|start| start..(start + batch_size).min(lengths.len()))
            .collect();
        assert_eq!(
            micro_batch_bounds(&lengths, batch_size, None),
            expected,
            "batch_size {batch_size}"
        );
    }
    assert!(micro_batch_bounds(&[], 16, None).is_empty());
    assert!(micro_batch_bounds(&[], 16, Some(2048)).is_empty());
}

#[test]
fn micro_batch_bounds_split_long_rows_by_token_budget() {
    let lengths = [10, 10, 10, 10, 100, 100, 100, 100];
    assert_eq!(
        micro_batch_bounds(&lengths, 16, Some(200)),
        vec![0..4, 4..6, 6..8]
    );
}

#[test]
fn micro_batch_bounds_keep_a_row_over_budget_alone() {
    assert_eq!(
        micro_batch_bounds(&[5, 5000, 6000], 16, Some(1000)),
        vec![0..1, 1..2, 2..3]
    );
}

#[test]
fn micro_batch_bounds_still_cap_rows_at_batch_size() {
    assert_eq!(
        micro_batch_bounds(&[1; 10], 4, Some(1000)),
        vec![0..4, 4..8, 8..10]
    );
}

#[test]
fn micro_batch_bounds_place_every_row_in_exactly_one_batch() {
    let lengths = [1, 2, 2, 7, 30, 31, 500, 900, 2000, 8192, 8192];
    for batch_size in [1, 2, 5, 16] {
        for budget in [None, Some(1), Some(64), Some(2048), Some(20_000)] {
            let bounds = micro_batch_bounds(&lengths, batch_size, budget);
            let mut next = 0;
            for range in &bounds {
                assert_eq!(range.start, next, "{batch_size} {budget:?}");
                let rows = range.len();
                assert!((1..=batch_size).contains(&rows), "{batch_size} {budget:?}");
                if let Some(budget) = budget {
                    assert!(
                        rows == 1 || rows * lengths[range.end - 1] <= budget,
                        "{batch_size} {budget:?} {range:?}"
                    );
                }
                next = range.end;
            }
            assert_eq!(next, lengths.len(), "{batch_size} {budget:?}");
        }
    }
}

#[test]
fn image_input_is_rejected_by_text_only_models() {
    let e = engine(8);
    let image = ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    };
    let err = e.embed_image(image, &EmbedOptions::default()).unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(ref m) if m.contains("image")));
}
