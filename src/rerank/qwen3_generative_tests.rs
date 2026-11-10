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

//! Qwen3 generative reranker gates.
//!
//! Four properties carry this path: the prompt is byte-identical to the
//! reference recipe, truncation never eats the assistant header the score is
//! read from, left padding puts every row's last real token at column
//! `L - 1`, and the score is the sigmoid of the yes/no logit difference at
//! that column.
//!
//! Everything runs on a synthetic 2-layer backbone and a word-level
//! tokenizer, so no checkpoint is needed; the real-checkpoint gate lives in
//! `real_checkpoint_tests.rs`.
//!
//! The last two tests write that backbone to a checkpoint directory and load
//! it through `load_reranker`, to pin when the load reads the weights.

use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;

use super::*;
use crate::embeddings::tokenize::{EncodedBatch, EncodedRow, PaddingSide};
use crate::models::embedding_test_support::{
    Rng, ids_array, max_abs_diff, mlx_test_guard, to_vec, write_f32_safetensors,
};
use crate::models::qwen3::ModelArgs;
use crate::rerank::{LoadedReranker, RerankLoadOptions, load_reranker_with_options};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 64;

/// Token ids the test tokenizer assigns; `yes` and `no` must be single tokens
/// for the score to be a difference of two logits.
const YES_ID: u32 = 5;
const NO_ID: u32 = 6;

fn tiny_args() -> ModelArgs {
    ModelArgs {
        model_type: "qwen3".to_string(),
        hidden_size: HIDDEN,
        num_hidden_layers: 2,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        num_key_value_heads: 1,
        head_dim: HEAD_DIM,
        max_position_embeddings: Some(512),
        rope_theta: 1_000_000.0,
        rope_scaling: None,
        checkpoint_label: None,
        tie_word_embeddings: true,
        quantization: None,
        quantization_config: None,
    }
}

/// Deterministic dense weights in the sanitized Qwen3 key layout.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    tiny_weights_seeded(args, 0x1356_0BAD_F00D)
}

/// [`tiny_weights`] drawn from `seed`, for two checkpoints that score apart.
fn tiny_weights_seeded(args: &ModelArgs, seed: u64) -> WeightMap {
    let mut rng = Rng::new(seed);
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
        for (key, shape) in [
            (format!("{p}.self_attn.q_proj.weight"), vec![q_out, h]),
            (format!("{p}.self_attn.k_proj.weight"), vec![kv_out, h]),
            (format!("{p}.self_attn.v_proj.weight"), vec![kv_out, h]),
            (format!("{p}.self_attn.o_proj.weight"), vec![h, q_out]),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, h]),
            (format!("{p}.mlp.up_proj.weight"), vec![inter, h]),
            (format!("{p}.mlp.down_proj.weight"), vec![h, inter]),
        ] {
            rng.insert(&mut w, &key, &shape, 0.2);
        }
        rng.insert(&mut w, &format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1);
        rng.insert(&mut w, &format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1);
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

/// A Qwen-shaped word-level tokenizer.
///
/// `extra_added` lets a test declare an added token that splits `yes` into
/// two pieces, which is the condition the yes/no guard rejects. The chat
/// markers are added tokens so the prompt scaffold encodes to a stable, small
/// number of ids without a real BPE vocabulary.
fn qwen_like_tokenizer(extra_added: Option<&str>) -> crate::tokenizer::MlxcelTokenizer {
    let json = qwen_like_tokenizer_json(extra_added);
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(json.as_bytes()).expect("qwen-like tokenizer parses");
    crate::tokenizer::MlxcelTokenizer::HuggingFace(tokenizer)
}

/// The `tokenizer.json` text behind [`qwen_like_tokenizer`].
fn qwen_like_tokenizer_json(extra_added: Option<&str>) -> String {
    let mut added = vec![
        r#"{"id": 0, "content": "<|endoftext|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}"#.to_string(),
        r#"{"id": 1, "content": "<|im_start|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}"#.to_string(),
        r#"{"id": 2, "content": "<|im_end|>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}"#.to_string(),
        r#"{"id": 3, "content": "<think>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}"#.to_string(),
        r#"{"id": 4, "content": "</think>", "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}"#.to_string(),
    ];
    if let Some(content) = extra_added {
        added.push(format!(
            r#"{{"id": 9, "content": "{content}", "special": false, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}}"#
        ));
    }
    format!(
        r#"{{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{}],
        "normalizer": null,
        "pre_tokenizer": {{"type": "Whitespace"}},
        "post_processor": null,
        "decoder": null,
        "model": {{
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {{
                "<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
                "<think>": 3, "</think>": 4, "yes": 5, "no": 6, "[UNK]": 7,
                "s": 8, "ye": 9, "panda": 10, "bear": 11, "china": 12,
                "capital": 13, "beijing": 14, "berlin": 15, "germany": 16
            }}
        }}
    }}"#,
        added.join(", ")
    )
}

/// Build a reranker over the synthetic backbone.
fn build(max_length: usize, batch_size: usize) -> Qwen3Reranker {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = crate::models::qwen3::Qwen3Model::from_weights(&weights, &args)
        .expect("synthetic Qwen3 loads");
    Qwen3Reranker::from_parts(
        model,
        &args,
        qwen_like_tokenizer(None),
        /* pad_token_id */ 0,
        max_length,
        batch_size,
    )
    .expect("reranker assembles")
}

#[test]
fn prompt_bytes_match_reference_recipe() {
    // Byte-for-byte the strings the Qwen3-Reranker model card publishes.
    assert_eq!(
        PROMPT_PREFIX,
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
         Query and the Instruct provided. Note that the answer can only be \"yes\" or \
         \"no\".<|im_end|>\n<|im_start|>user\n"
    );
    assert_eq!(
        PROMPT_SUFFIX,
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(
        DEFAULT_INSTRUCTION,
        "Given a web search query, retrieve relevant passages that answer the query"
    );
    assert_eq!(
        prompt_content("Find it", "what is panda?", "a bear"),
        "<Instruct>: Find it\n<Query>: what is panda?\n<Document>: a bear"
    );
    assert_eq!(
        resolve_instruction(None),
        DEFAULT_INSTRUCTION,
        "an absent instruction falls back to the card's default"
    );
    assert_eq!(
        resolve_instruction(Some("   ")),
        DEFAULT_INSTRUCTION,
        "a blank instruction is not an instruction"
    );
    assert_eq!(resolve_instruction(Some(" custom ")), "custom");
}

#[test]
fn content_truncation_keeps_prefix_and_suffix() {
    let _guard = mlx_test_guard();
    // The scaffold is fixed, so size the limit relative to it rather than to
    // a literal: the word-level test tokenizer spends more ids on the system
    // turn than the real BPE vocabulary would.
    let measured = build(4096, 4);
    let scaffold = measured.prefix_ids.len() + measured.suffix_ids.len();
    let budget = 24;
    let reranker = build(scaffold + budget, 4);

    let long_document = "panda bear china ".repeat(200);
    let ids = reranker
        .encode_row(DEFAULT_INSTRUCTION, "what is panda?", &long_document)
        .expect("row encodes");
    assert_eq!(
        ids.len(),
        scaffold + budget,
        "an over-long pair fills the limit exactly"
    );
    assert!(
        ids.starts_with(&reranker.prefix_ids),
        "the system prefix must survive truncation"
    );
    assert!(
        ids.ends_with(&reranker.suffix_ids),
        "the assistant header the score is read from must survive truncation"
    );

    // A pair that fits inside the budget is left alone.
    let short = reranker
        .encode_row("find", "panda", "bear")
        .expect("row encodes");
    assert!(
        short.len() < scaffold + budget,
        "a short pair must not be padded up to the limit, got {}",
        short.len()
    );
    assert!(short.starts_with(&reranker.prefix_ids));
    assert!(short.ends_with(&reranker.suffix_ids));

    // The scaffold alone must leave room, or the load fails loudly instead of
    // silently scoring an empty pair.
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = crate::models::qwen3::Qwen3Model::from_weights(&weights, &args)
        .expect("synthetic Qwen3 loads");
    let err = match Qwen3Reranker::from_parts(model, &args, qwen_like_tokenizer(None), 0, 4, 4) {
        Ok(_) => panic!("a limit smaller than the scaffold must be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("leaves no room"), "{err}");
}

#[test]
fn left_padding_puts_last_token_at_l_minus_1() {
    let rows = vec![
        EncodedRow {
            ids: vec![10, 11, 12],
            type_ids: None,
        },
        EncodedRow {
            ids: vec![20, 21, 22, 23, 24],
            type_ids: None,
        },
    ];
    let batch = EncodedBatch::from_rows_with_padding(&rows, 0, None, PaddingSide::Left);
    assert_eq!(batch.width, 5);
    assert_eq!(batch.token_counts, vec![3, 5]);
    assert_eq!(batch.input_ids, vec![0, 0, 10, 11, 12, 20, 21, 22, 23, 24]);
    assert_eq!(batch.attention_mask, vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);
    for (row, expected_last) in [(0usize, 12i32), (1, 24)] {
        assert_eq!(
            batch.input_ids[row * batch.width + batch.width - 1],
            expected_last,
            "row {row} must end on its own last real token"
        );
    }

    // Right padding is still the default and still puts the padding after the
    // real tokens, which is what every encoder family relies on.
    let right = EncodedBatch::from_rows(&rows, 0, None);
    assert_eq!(right.input_ids, vec![10, 11, 12, 0, 0, 20, 21, 22, 23, 24]);
    assert_eq!(right.attention_mask, vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1]);
}

#[test]
fn yes_no_ids_must_be_single_tokens() {
    let _guard = mlx_test_guard();
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = crate::models::qwen3::Qwen3Model::from_weights(&weights, &args)
        .expect("synthetic Qwen3 loads");
    // The added token `ye` splits `yes` into `ye` + `s`, so the score would no
    // longer be a difference of two logits.
    let err =
        match Qwen3Reranker::from_parts(model, &args, qwen_like_tokenizer(Some("ye")), 0, 128, 4) {
            Ok(_) => panic!("a split answer token must be rejected"),
            Err(err) => err,
        };
    assert!(
        err.to_string().contains("single token"),
        "unexpected message: {err}"
    );

    // The plain tokenizer resolves both to the ids the vocab declares.
    let reranker = build(128, 4);
    assert_eq!(reranker.yes_id, YES_ID);
    assert_eq!(reranker.no_id, NO_ID);
}

#[test]
fn sigmoid_of_logit_difference() {
    let _guard = mlx_test_guard();
    let reranker = build(128, 4);
    let documents = [
        RerankItem::text("panda bear china"),
        RerankItem::text("berlin germany"),
    ];
    let scored = reranker
        .score(&RerankItem::text("what is panda?"), &documents, None)
        .expect("scoring succeeds");
    assert_eq!(scored.scores.len(), 2);
    assert!(
        scored
            .scores
            .iter()
            .all(|s| s.is_finite() && (0.0..=1.0).contains(s)),
        "scores must be probabilities, got {:?}",
        scored.scores
    );
    assert!(scored.prompt_tokens > 0);

    // Recompute the first row's score by hand from the same backbone: prompt
    // ids, one prefill, the last column's logits, sigmoid of yes minus no.
    let ids = reranker
        .encode_row(DEFAULT_INSTRUCTION, "what is panda?", "panda bear china")
        .expect("row encodes");
    let signed: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let length = signed.len() as i32;
    let input = ids_array(&signed, 1, length);
    let mask = ids_array(&vec![1; signed.len()], 1, length);
    let causal = mlxcel_core::utils::create_causal_padding_mask(&mask, 0);
    let mut caches = reranker.model.make_caches();
    let hidden = reranker
        .model
        .forward_hidden(&input, None, &mut caches, Some(&causal));
    let last = slice_axis(&hidden, 1, length - 1, length);
    let logits = to_vec(&reranker.model.embed_tokens.as_linear(&last));
    let expected = 1.0 / (1.0 + (-(logits[YES_ID as usize] - logits[NO_ID as usize])).exp());

    // The batched call left-pads the shorter row, which shifts its RoPE
    // positions, so only the row that is already the longest is comparable
    // element for element. Score it alone to make the comparison exact.
    let alone = reranker
        .score(
            &RerankItem::text("what is panda?"),
            std::slice::from_ref(&documents[0]),
            None,
        )
        .expect("single-document scoring succeeds");
    assert!(
        (alone.scores[0] - expected).abs() < 1e-5,
        "score {} does not match the hand-computed sigmoid {expected}",
        alone.scores[0]
    );
}

#[test]
fn rejects_image_items() {
    let _guard = mlx_test_guard();
    let reranker = build(128, 4);
    let image = crate::rerank::ImageInput {
        image: image::DynamicImage::new_rgb8(8, 8),
    };
    let err = reranker
        .score(
            &RerankItem::text("query"),
            &[RerankItem::image(image)],
            None,
        )
        .expect_err("a text-only reranker rejects images");
    assert!(err.to_string().contains("text-only"), "{err}");
}

#[test]
fn score_until_stops_between_micro_batches_when_abandoned() {
    let _guard = mlx_test_guard();
    let reranker = build(128, 2);
    let documents = [
        RerankItem::text("panda bear china"),
        RerankItem::text("berlin germany"),
        RerankItem::text("capital beijing"),
        RerankItem::text("panda china"),
    ];
    let checks = std::cell::Cell::new(0usize);
    let err = reranker
        .score_until(
            &RerankItem::text("what is panda?"),
            &documents,
            None,
            &|| {
                let n = checks.get();
                checks.set(n + 1);
                n >= 1
            },
        )
        .expect_err("abandons before the second micro-batch");
    assert!(err.is::<crate::rerank::RerankAbandoned>(), "{err}");
    assert_eq!(checks.get(), 2, "checked before batch 0 and before batch 1");
}

/// Positions the forward passes compute when `order` is cut into micro-batches
/// of `batch_size`: every batch is padded to its longest row.
fn computed_positions(lengths: &[usize], order: &[usize], batch_size: usize) -> usize {
    order
        .chunks(batch_size)
        .map(|chunk| {
            let width = chunk.iter().map(|&index| lengths[index]).max().unwrap_or(0);
            width * chunk.len()
        })
        .sum()
}

#[test]
fn batching_order_puts_rows_of_similar_length_in_one_micro_batch() {
    let lengths = [3, 9, 2, 8, 1, 7];
    let arrival: Vec<usize> = (0..lengths.len()).collect();
    assert_eq!(
        computed_positions(&lengths, &arrival, 2),
        48,
        "arrival order pads every short row to a long one: 2*9 + 2*8 + 2*7"
    );

    let order = batching_order(&lengths);
    assert_eq!(order, vec![1, 3, 5, 0, 2, 4], "longest row first");
    assert_eq!(
        computed_positions(&lengths, &order, 2),
        36,
        "length-sorted batches are 2*9 + 2*7 + 2*2"
    );
}

#[test]
fn batching_order_keeps_arrival_order_among_equal_lengths() {
    assert_eq!(batching_order(&[4, 6, 4, 6, 4]), vec![1, 3, 0, 2, 4]);
    assert_eq!(batching_order(&[5, 5, 5]), vec![0, 1, 2]);
    assert_eq!(batching_order(&[7]), vec![0]);
    assert!(batching_order(&[]).is_empty());
}

/// Documents of five different lengths, in an order that is not sorted by
/// length in either direction.
fn documents_of_unsorted_lengths() -> Vec<RerankItem> {
    vec![
        RerankItem::text("berlin"),
        RerankItem::text("panda bear china ".repeat(6)),
        RerankItem::text("capital beijing"),
        RerankItem::text("germany berlin ".repeat(3)),
        RerankItem::text("panda china bear"),
    ]
}

/// Token count of the prompt row each document encodes to.
fn row_lengths(reranker: &Qwen3Reranker, query: &str, documents: &[RerankItem]) -> Vec<usize> {
    documents
        .iter()
        .map(|document| {
            let text = document.text_or_empty();
            reranker
                .encode_row(DEFAULT_INSTRUCTION, query, text)
                .expect("row encodes")
                .len()
        })
        .collect()
}

#[test]
fn scores_come_back_at_each_document_s_own_index() {
    let _guard = mlx_test_guard();
    let reranker = build(256, 2);
    let query = RerankItem::text("what is panda?");
    let documents = documents_of_unsorted_lengths();

    let lengths = row_lengths(&reranker, "what is panda?", &documents);
    let arrival: Vec<usize> = (0..documents.len()).collect();
    assert_ne!(
        batching_order(&lengths),
        arrival,
        "the request must be one the reranker reorders, lengths {lengths:?}"
    );

    // A document scored alone is never padded, so it is the reference for
    // its own index whatever batch it lands in.
    let alone: Vec<f32> = documents
        .iter()
        .map(|document| {
            reranker
                .score(&query, std::slice::from_ref(document), None)
                .expect("single-document scoring succeeds")
                .scores[0]
        })
        .collect();
    for (i, a) in alone.iter().enumerate() {
        for (j, b) in alone.iter().enumerate().skip(i + 1) {
            assert!(
                (a - b).abs() > 5e-6,
                "documents {i} and {j} must score apart for a misplaced score to show: {alone:?}"
            );
        }
    }

    let scored = reranker
        .score(&query, &documents, None)
        .expect("scoring succeeds");
    assert_eq!(scored.scores.len(), documents.len());
    assert!(
        max_abs_diff(&scored.scores, &alone) < 1e-6,
        "scores {:?} are not in document order, each document alone scores {alone:?}",
        scored.scores
    );
}

#[test]
fn prompt_tokens_count_every_row_once_whatever_the_batching_order() {
    let _guard = mlx_test_guard();
    let reranker = build(256, 2);
    let documents = documents_of_unsorted_lengths();
    let expected: usize = row_lengths(&reranker, "what is panda?", &documents)
        .iter()
        .sum();

    let scored = reranker
        .score(&RerankItem::text("what is panda?"), &documents, None)
        .expect("scoring succeeds");
    assert_eq!(scored.prompt_tokens, expected);
}

const CHECKPOINT_SEED: u64 = 0x1356_0BAD_F00D;
const REWRITTEN_SEED: u64 = 0x0DD_C0FF_EE11;

/// The tensors of a synthetic checkpoint, in the safetensors writer's shape.
fn checkpoint_tensors(seed: u64) -> Vec<(String, Vec<i32>, Vec<f32>)> {
    tiny_weights_seeded(&tiny_args(), seed)
        .iter()
        .map(|(name, array)| (name.clone(), mlxcel_core::array_shape(array), to_vec(array)))
        .collect()
}

/// Write a checkpoint directory `load_reranker` accepts as a Qwen3 reranker.
fn write_checkpoint(dir: &std::path::Path, seed: u64) {
    let args = tiny_args();
    let config = serde_json::json!({
        "model_type": args.model_type,
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "intermediate_size": args.intermediate_size,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "rms_norm_eps": args.rms_norm_eps,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": args.rope_theta,
        "tie_word_embeddings": args.tie_word_embeddings,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("config written");
    std::fs::write(dir.join("tokenizer.json"), qwen_like_tokenizer_json(None))
        .expect("tokenizer written");
    write_f32_safetensors(&dir.join("model.safetensors"), &checkpoint_tensors(seed));
}

/// Replace the tensor values of a written checkpoint without replacing the
/// file, so a tensor that is read from it after this call gets the new values.
fn rewrite_checkpoint_values(dir: &std::path::Path, seed: u64) {
    use std::io::Write;

    let staging = tempfile::tempdir().expect("tempdir");
    let staged = staging.path().join("model.safetensors");
    write_f32_safetensors(&staged, &checkpoint_tensors(seed));
    let bytes = std::fs::read(&staged).expect("staged checkpoint reads");
    let path = dir.join("model.safetensors");
    assert_eq!(
        bytes.len() as u64,
        std::fs::metadata(&path).expect("checkpoint exists").len(),
        "the rewrite must keep the file layout"
    );
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("checkpoint opens for writing");
    file.write_all(&bytes).expect("checkpoint rewritten");
    file.sync_all().expect("checkpoint synced");
}

fn score_fixed_request(loaded: &LoadedReranker) -> Vec<f32> {
    loaded
        .reranker
        .score(
            &RerankItem::text("what is panda?"),
            &[
                RerankItem::text("panda bear china"),
                RerankItem::text("berlin germany"),
            ],
            None,
        )
        .expect("scoring succeeds")
        .scores
}

/// Scores around a checkpoint rewrite.
struct RewriteScores {
    /// A reranker loaded from the original checkpoint and scored at once.
    original: Vec<f32>,
    /// A reranker loaded by `load` from the original checkpoint, but scored
    /// only after the file was rewritten.
    held_across_rewrite: Vec<f32>,
    /// A reranker loaded from the rewritten checkpoint.
    rewritten: Vec<f32>,
}

/// Load with `load`, rewrite the checkpoint under the loaded reranker, then
/// score. Whether the load read the weights shows in which checkpoint the
/// held reranker scores like.
fn scores_around_a_rewrite(
    load: impl Fn(&std::path::Path) -> anyhow::Result<LoadedReranker>,
) -> RewriteScores {
    let dir = tempfile::tempdir().expect("tempdir");
    write_checkpoint(dir.path(), CHECKPOINT_SEED);
    let default_load =
        |path: &std::path::Path| load_reranker_with_options(path, RerankLoadOptions::default());

    let original = score_fixed_request(&default_load(dir.path()).expect("checkpoint loads"));
    let held = load(dir.path()).expect("checkpoint loads");
    rewrite_checkpoint_values(dir.path(), REWRITTEN_SEED);
    let held_across_rewrite = score_fixed_request(&held);
    let rewritten = score_fixed_request(&default_load(dir.path()).expect("checkpoint loads"));
    assert!(
        max_abs_diff(&original, &rewritten) > 1e-3,
        "the two checkpoints must score apart for the comparison to mean anything: \
         {original:?} against {rewritten:?}"
    );
    RewriteScores {
        original,
        held_across_rewrite,
        rewritten,
    }
}

#[test]
fn load_reads_the_weights_before_it_returns() {
    let _guard = mlx_test_guard();
    let scores = scores_around_a_rewrite(|path| {
        load_reranker_with_options(path, RerankLoadOptions::default())
    });
    assert!(
        max_abs_diff(&scores.held_across_rewrite, &scores.original) < 1e-6,
        "the first scoring call read the checkpoint: scores {:?} follow the rewritten file \
         ({:?}) instead of the one that was loaded ({:?})",
        scores.held_across_rewrite,
        scores.rewritten,
        scores.original
    );
}

#[test]
fn lazy_policy_leaves_the_weights_to_the_first_scoring_call() {
    use mlxcel_core::weight_materialize::MaterializationPolicy;

    let _guard = mlx_test_guard();
    let scores = scores_around_a_rewrite(|path| {
        crate::rerank::loader::load_reranker_with_policy(
            path,
            RerankLoadOptions::default(),
            MaterializationPolicy::Lazy,
        )
    });
    assert!(
        max_abs_diff(&scores.held_across_rewrite, &scores.rewritten) < 1e-6,
        "a lazy load read the checkpoint before the first scoring call: scores {:?} follow \
         the file that was loaded ({:?}) instead of the rewritten one ({:?})",
        scores.held_across_rewrite,
        scores.original,
        scores.rewritten
    );
}
