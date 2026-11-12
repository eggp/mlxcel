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

//! Qwen3-Embedding gates.
//!
//! Three properties carry this port. The `forward_hidden` split must not
//! change generation, so the first test reassembles the head on top of it and
//! demands a token-exact match with `forward_impl`. The backbone must stay
//! causal, unlike EmbeddingGemma's. And last-token pooling must pick the
//! appended `<|endoftext|>` in a right-padded batch, which is the whole reason
//! padding can sit after the pooled position.
//!
//! The last test loads the real `Qwen/Qwen3-Embedding-0.6B` checkpoint and
//! soft-skips when it is not downloaded.

use anyhow::Result;
use mlxcel_core::utils::create_causal_padding_mask;
use mlxcel_core::weights::WeightMap;

use super::Qwen3EmbeddingModel;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::{PoolingMode, pool};
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, max_abs_diff, mlx_test_guard, to_vec,
};
use crate::models::qwen3::{ModelArgs, Qwen3Model};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;

const QWEN3_EMBEDDING: &str = "Qwen/Qwen3-Embedding-0.6B";

/// The reference query-document cosine matrix from the checkpoint's model
/// card, for the two queries and two documents below. The card publishes
/// these for the bf16 original; no PyTorch or `transformers` install is
/// available on the validation host, so this is the only reference the gate
/// can compare against.
const MODEL_CARD_SIMILARITY: [[f32; 2]; 2] = [[0.7646, 0.1414], [0.1355, 0.6000]];
/// Tolerance the issue sets for the model-card matrix.
const MODEL_CARD_TOLERANCE: f32 = 2e-2;

fn tiny_args(num_layers: usize) -> ModelArgs {
    ModelArgs {
        model_type: "qwen3".to_string(),
        hidden_size: HIDDEN,
        num_hidden_layers: num_layers,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        num_key_value_heads: 1,
        head_dim: HEAD_DIM,
        max_position_embeddings: Some(128),
        rope_theta: 1_000_000.0,
        rope_scaling: None,
        checkpoint_label: None,
        tie_word_embeddings: true,
        quantization: None,
        quantization_config: None,
    }
}

/// Deterministic dense weights for `args`, in the sanitized key layout.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut rng = Rng::new(0x00C0_FFEE_1329);
    let mut w = WeightMap::new();
    let h = args.hidden_size as i32;
    let hd = args.head_dim as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_key_value_heads as i32 * hd;
    let inter = args.intermediate_size as i32;

    rng.insert(
        &mut w,
        "model.embed_tokens.weight",
        &[args.vocab_size as i32, h],
        0.5,
    );
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.q_proj.weight"),
            &[q_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv_out, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.self_attn.o_proj.weight"),
            &[h, q_out],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1);
        rng.insert(&mut w, &format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1);
        rng.insert(
            &mut w,
            &format!("{p}.mlp.gate_proj.weight"),
            &[inter, h],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.mlp.up_proj.weight"), &[inter, h], 0.2);
        rng.insert(
            &mut w,
            &format!("{p}.mlp.down_proj.weight"),
            &[h, inter],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.input_layernorm.weight"), &[h], 0.1);
        rng.insert(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            0.1,
        );
    }
    rng.insert(&mut w, "model.norm.weight", &[h], 0.1);
    w
}

fn build(args: &ModelArgs, weights: &WeightMap) -> Result<Qwen3EmbeddingModel> {
    Qwen3EmbeddingModel::from_weights(weights, args, PoolingMode::LastToken, true)
}

fn ramp(len: usize) -> Vec<i32> {
    (0..len).map(|i| (i % VOCAB) as i32).collect()
}

/// `[L, HIDDEN]` hidden states of one unpadded row.
fn hidden_states(model: &Qwen3EmbeddingModel, ids: &[i32]) -> Vec<f32> {
    let input = ids_array(ids, 1, ids.len() as i32);
    to_vec(&model.forward_hidden(&input))
}

/// Pooled `[B, HIDDEN]` vectors of one `[B, width]` batch through `embed`.
fn pooled(model: &Qwen3EmbeddingModel, ids: &[i32], mask: &[i32], width: usize) -> Vec<f32> {
    let batch = (ids.len() / width) as i32;
    let input_ids = ids_array(ids, batch, width as i32);
    let attention_mask = ids_array(mask, batch, width as i32);
    to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &input_ids,
                attention_mask: &attention_mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    )
}

// Backbone refactor.

#[test]
fn forward_hidden_then_head_matches_forward_impl() {
    let _guard = mlx_test_guard();
    // The split must be a pure refactor: applying the (tied) head to
    // `forward_hidden` has to reproduce `forward_impl` bit for bit.
    let args = tiny_args(2);
    let weights = tiny_weights(&args);
    let model = Qwen3Model::from_weights(&weights, &args).expect("synthetic Qwen3 loads");

    let ids = ramp(12);
    let input = ids_array(&ids, 1, ids.len() as i32);

    let mut caches = model.make_caches();
    let logits = to_vec(&model.forward_impl(&input, None, &mut caches, None));

    let mut fresh = model.make_caches();
    let hidden = model.forward_hidden(&input, None, &mut fresh, None);
    let reassembled = to_vec(&model.embed_tokens.as_linear(&hidden));

    assert_eq!(logits.len(), reassembled.len());
    assert_eq!(
        max_abs_diff(&logits, &reassembled),
        0.0,
        "forward_hidden plus the head is not token-exact with forward_impl"
    );
}

// Forward pass.

#[test]
fn causal_prefill_is_causal() {
    let _guard = mlx_test_guard();
    // The mirror image of the EmbeddingGemma bidirectionality gate: flipping
    // the last token must leave every earlier hidden state untouched.
    let args = tiny_args(2);
    let model = build(&args, &tiny_weights(&args)).unwrap();

    let mut ids = ramp(96);
    let baseline = hidden_states(&model, &ids);
    ids[95] = (ids[95] + 7) % VOCAB as i32;
    let perturbed = hidden_states(&model, &ids);

    let earlier = 95 * HIDDEN;
    let moved = max_abs_diff(&baseline[..earlier], &perturbed[..earlier]);
    assert!(
        moved < 1e-6,
        "an earlier token reacted to the last token ({moved}); the forward pass is not causal"
    );
    let last = max_abs_diff(&baseline[earlier..], &perturbed[earlier..]);
    assert!(
        last > 1e-4,
        "the changed token itself did not move ({last})"
    );
}

#[test]
fn last_token_pool_uses_appended_eos() {
    let _guard = mlx_test_guard();
    // Right padding after the pooled position must not change the vector: the
    // padded row's last real token sees exactly the keys the unpadded row does.
    let args = tiny_args(2);
    let model = build(&args, &tiny_weights(&args)).unwrap();

    let short = ramp(8);
    let long: Vec<i32> = (0..12).map(|i| ((i * 5 + 3) % VOCAB) as i32).collect();

    let alone = ids_array(&short, 1, short.len() as i32);
    let alone_mask = ids_array(&vec![1; short.len()], 1, short.len() as i32);
    let alone_values = to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &alone,
                attention_mask: &alone_mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    );

    let mut padded_ids = short.clone();
    padded_ids.resize(12, 0);
    padded_ids.extend_from_slice(&long);
    let mut padded_mask = vec![1; short.len()];
    padded_mask.resize(12, 0);
    padded_mask.extend(std::iter::repeat_n(1, 12));

    let batch_ids = ids_array(&padded_ids, 2, 12);
    let batch_mask = ids_array(&padded_mask, 2, 12);
    let batch_values = to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &batch_ids,
                attention_mask: &batch_mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    );

    assert_eq!(alone_values.len(), HIDDEN);
    assert_eq!(batch_values.len(), 2 * HIDDEN);
    assert!(max_abs_diff(&alone_values, &batch_values[..HIDDEN]) < 1e-4);
}

#[test]
fn mask_free_batch_matches_solo_rows_and_the_masked_path() {
    let _guard = mlx_test_guard();
    // The forward pass builds no mask array. In a right-padded batch every
    // real row must still get the vector it gets alone, and the vector the
    // explicit causal-and-padding mask gives.
    const WIDTH: usize = 12;
    // Rounding only: the three runs are the same f32 math through different
    // batch shapes and attention routes. Measured on Metal, the spread is at
    // most 1.5e-8 on components of magnitude up to 0.14, so the tolerance
    // leaves room for another backend's rounding. A padding key that reaches
    // a real row moves its vector by at least 2.2e-3, which the left-padded
    // control below pins, so the tolerance cannot hide a leak.
    const TOLERANCE: f32 = 1e-5;
    const LEAK: f32 = 1e-3;

    let args = tiny_args(2);
    let model = build(&args, &tiny_weights(&args)).unwrap();

    // One token, two partly padded rows, and one row that fills the width.
    let rows: Vec<Vec<i32>> = [1_usize, 5, 9, WIDTH]
        .iter()
        .enumerate()
        .map(|(r, &len)| {
            (0..len)
                .map(|i| ((i * (r + 3) + 2 * r + 1) % VOCAB) as i32)
                .collect()
        })
        .collect();

    let mut right_ids = Vec::new();
    let mut right_mask = Vec::new();
    let mut left_ids = Vec::new();
    let mut left_mask = Vec::new();
    for row in &rows {
        let padding = WIDTH - row.len();
        right_ids.extend_from_slice(row);
        right_ids.extend(std::iter::repeat_n(0, padding));
        right_mask.extend(std::iter::repeat_n(1, row.len()));
        right_mask.extend(std::iter::repeat_n(0, padding));
        left_ids.extend(std::iter::repeat_n(0, padding));
        left_ids.extend_from_slice(row);
        left_mask.extend(std::iter::repeat_n(0, padding));
        left_mask.extend(std::iter::repeat_n(1, row.len()));
    }

    let batch = pooled(&model, &right_ids, &right_mask, WIDTH);

    let batch_ids = ids_array(&right_ids, rows.len() as i32, WIDTH as i32);
    let batch_mask = ids_array(&right_mask, rows.len() as i32, WIDTH as i32);
    let mut caches = model.model.make_caches();
    let masked_hidden = model.model.forward_hidden(
        &batch_ids,
        None,
        &mut caches,
        Some(&create_causal_padding_mask(&batch_mask, 0)),
    );
    let masked = to_vec(&pool(&masked_hidden, &batch_mask, PoolingMode::LastToken));

    let left = pooled(&model, &left_ids, &left_mask, WIDTH);

    assert_eq!(batch.len(), rows.len() * HIDDEN);
    assert_eq!(masked.len(), batch.len());
    for (r, row) in rows.iter().enumerate() {
        let solo = pooled(&model, row, &vec![1; row.len()], row.len());
        let span = r * HIDDEN..(r + 1) * HIDDEN;

        let vs_solo = max_abs_diff(&solo, &batch[span.clone()]);
        assert!(
            vs_solo < TOLERANCE,
            "row {r} ({} tokens) moved by {vs_solo} between the solo and the padded run",
            row.len()
        );
        let vs_masked = max_abs_diff(&masked[span.clone()], &batch[span.clone()]);
        assert!(
            vs_masked < TOLERANCE,
            "row {r} ({} tokens) differs by {vs_masked} from the masked forward pass",
            row.len()
        );

        // Left padding puts padding keys in front of the real tokens, where
        // causal attention alone does not hide them.
        let vs_left = max_abs_diff(&solo, &left[span]);
        if row.len() < WIDTH {
            assert!(
                vs_left > LEAK,
                "row {r} ({} tokens) moved by only {vs_left} under left padding; the comparison cannot see a padding leak",
                row.len()
            );
        }
        eprintln!(
            "GATE mask-free row={r} tokens={} vs_solo={vs_solo:.3e} vs_masked={vs_masked:.3e} vs_left={vs_left:.3e}",
            row.len()
        );
    }
}

#[test]
fn instruction_wraps_only_when_supplied() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1);
    let model = build(&args, &tiny_weights(&args)).unwrap();
    assert_eq!(model.format_text("Paris", None), "Paris");
    assert_eq!(model.format_text("Paris", Some("   ")), "Paris");
    assert_eq!(
        model.format_text(
            "What is the capital of China?",
            Some("Given a web search query, retrieve relevant passages that answer the query")
        ),
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?"
    );
    assert_eq!(model.default_pooling(), PoolingMode::LastToken);
    assert_eq!(model.embedding_dim(), HIDDEN);
}

#[test]
fn images_are_rejected() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1);
    let model = build(&args, &tiny_weights(&args)).unwrap();
    let ids = ids_array(&[1, 2, 3], 1, 3);
    let mask = ids_array(&[1, 1, 1], 1, 3);
    let images = [crate::embeddings::model::ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    }];
    let err = err_string(model.embed(&EmbeddingBatch {
        input_ids: &ids,
        attention_mask: &mask,
        token_type_ids: None,
        images: Some(&images),
    }));
    assert!(err.contains("text-only"), "{err}");
}

// Real checkpoint.

#[test]
fn qwen3_embedding_checkpoint_matches_the_model_card_similarity_matrix() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(QWEN3_EMBEDDING) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("Qwen3-Embedding loads");
    assert_eq!(loaded.model_type, crate::models::ModelType::Qwen3Embedding);
    assert_eq!(loaded.limits.dim, 1024);

    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 16);
    const TASK: &str = "Given a web search query, retrieve relevant passages that answer the query";
    // Index 4 repeats index 0 verbatim: two byte-identical rows of one
    // right-padded batch must produce cosine 1.0.
    let texts: Vec<String> = [
        format!("Instruct: {TASK}\nQuery: What is the capital of China?"),
        format!("Instruct: {TASK}\nQuery: Explain gravity"),
        "The capital of China is Beijing.".to_string(),
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".to_string(),
        format!("Instruct: {TASK}\nQuery: What is the capital of China?"),
    ]
    .to_vec();
    let reply = engine
        .embed_texts(&texts, &crate::embeddings::EmbedOptions::default())
        .expect("Qwen3-Embedding embeds");

    assert_eq!(reply.vectors.len(), 5);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![1024]);
        assert!(
            vector.values.iter().all(|v| v.is_finite()),
            "a non-finite component reached the caller"
        );
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit vector, got {norm}");
    }
    let duplicate = cosine(&reply.vectors[0].values, &reply.vectors[4].values);
    assert!(
        (duplicate - 1.0).abs() < 1e-6,
        "identical inputs scored {duplicate}, not 1.0"
    );
    let mut observed = [[0.0_f32; 2]; 2];
    for (q, expected_row) in MODEL_CARD_SIMILARITY.iter().enumerate() {
        for (d, expected) in expected_row.iter().enumerate() {
            let got = cosine(&reply.vectors[q].values, &reply.vectors[2 + d].values);
            observed[q][d] = got;
            assert!(
                (got - expected).abs() < MODEL_CARD_TOLERANCE,
                "query {q} vs document {d}: got {got}, model card says {expected}"
            );
        }
    }

    // Right padding sits after the pooled position, so the padded row must
    // reproduce the solo run. This is the property causal attention plus
    // last-token pooling buys, on the real checkpoint rather than a synthetic.
    let solo = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("Qwen3-Embedding embeds a single row");
    let drift = max_abs_diff(&solo.vectors[0].values, &reply.vectors[2].values);
    assert!(
        drift < 1e-3,
        "the document drifted by {drift} between the solo and the padded run"
    );

    // Printed so a repeated run shows the spread rather than one pass/fail bit.
    eprintln!(
        "GATE Qwen3-Embedding-0.6B duplicate={duplicate:.9} matrix=[[{:.6}, {:.6}], [{:.6}, {:.6}]] \
         batch_drift={drift:.3e}",
        observed[0][0], observed[0][1], observed[1][0], observed[1][1]
    );
}
