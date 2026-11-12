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

//! The Qwen3 generative reranker: an ordinary `Qwen3ForCausalLM` checkpoint
//! asked a yes/no question and read at one position.
//!
//! `Qwen/Qwen3-Reranker-*` ships no classification head. The reference recipe
//! wraps the pair in a fixed chat prompt that ends on the assistant header,
//! runs one prefill, and turns the two candidate answers into a probability:
//!
//! ```text
//! score = sigmoid(logits[L - 1, id("yes")] - logits[L - 1, id("no")])
//! ```
//!
//! The prompt is assembled from token ids, not from one formatted string, so
//! truncation can shorten the pair without ever touching the prefix or the
//! assistant header the score is read from.
//!
//! The rows of one request all open with the same tokens: the system turn, the
//! instruction and the query. A request of several documents forwards that
//! shared run once, at batch 1, and keeps each layer's keys and values. Each
//! micro-batch then forwards only what its rows hold after the run, against a
//! cache that starts as the run repeated per row:
//!
//! - a tail continues right after the run, at the positions it has in a single
//!   full row, so these batches are **right**-padded and each score is read at
//!   the row's own last real token;
//! - the causal mask hides the trailing padding from every real token, and the
//!   run is visible to all of them.
//!
//! A request with one document has nothing to share, so its row is forwarded
//! whole, and two details make that read correct:
//!
//! - rows forwarded whole are **left**-padded, which is what puts every row's
//!   last real token at column `L - 1`. Right padding would read the score off
//!   a pad token for every row but the longest;
//! - the mask is [`create_causal_padding_mask`], so the pad prefix is blocked
//!   as a key everywhere and the leading padding rows keep a finite softmax.
//!
//! Absolute positions still run `0..L` over the padded row, exactly as the
//! reference `Qwen3ForCausalLM.forward(input_ids, attention_mask)` does when it
//! derives `position_ids` from the cache position: the pad prefix shifts every
//! real token by the same amount, and RoPE only sees position differences among
//! the keys that are not masked out.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::layers::KVCache;
use mlxcel_core::utils::{create_causal_padding_mask, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::embeddings::limits::{derive_max_length, resolve_pad_token_id};
use crate::embeddings::tokenize::{EncodedBatch, EncodedRow, PaddingSide, truncate_token_ids};
use crate::models::qwen3::{ModelArgs, Qwen3Model};
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::{
    RERANK_MAX_LENGTH_CAP, RerankItem, RerankScores, Reranker, RerankerKind, sigmoid_to_vec,
};

/// System turn and the opening of the user turn. Byte-identical to the
/// reference recipe published on the `Qwen/Qwen3-Reranker-*` model cards.
pub const PROMPT_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";

/// Close of the user turn plus the assistant header with an empty think block.
/// The last token of this run is the position the score is read from.
pub const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Task description used when the request carries no `instruction`.
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Render the variable middle of the prompt.
#[must_use]
pub fn prompt_content(instruction: &str, query: &str, document: &str) -> String {
    format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}")
}

/// The task description a request actually renders with: its own
/// `instruction` when it carries a non-blank one, otherwise
/// [`DEFAULT_INSTRUCTION`].
#[must_use]
pub fn resolve_instruction(instruction: Option<&str>) -> &str {
    instruction
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .unwrap_or(DEFAULT_INSTRUCTION)
}

/// The Qwen3 yes/no reranker.
pub struct Qwen3Reranker {
    model: Qwen3Model,
    tokenizer: MlxcelTokenizer,
    yes_id: u32,
    no_id: u32,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
    pad_token_id: u32,
    max_length: usize,
    batch_size: usize,
}

/// Resolve the single token id a yes/no answer encodes to.
///
/// The score is a difference of two logits, so each answer has to be exactly
/// one token; a tokenizer that splits either word would make the read
/// meaningless rather than merely inaccurate.
fn single_token_id(tokenizer: &MlxcelTokenizer, word: &str) -> Result<u32> {
    let ids = tokenizer
        .encode(word, false)
        .with_context(|| format!("failed to encode the reranker answer token `{word}`"))?;
    match ids.as_slice() {
        [id] => Ok(*id),
        other => bail!(
            "the Qwen3 reranker needs `{word}` to be a single token, this tokenizer encodes it \
             as {other:?}"
        ),
    }
}

impl Qwen3Reranker {
    /// Load a Qwen3 reranker checkpoint from its directory.
    pub fn load(
        model_dir: &Path,
        batch_size: usize,
        max_length_override: Option<usize>,
    ) -> Result<Self> {
        let (model, args) = Qwen3Model::load(model_dir)
            .map_err(|e| anyhow::anyhow!("failed to load {} as Qwen3: {e}", model_dir.display()))?;
        let tokenizer = crate::embeddings::tokenize::strip_padding_and_truncation(
            load_tokenizer(model_dir).with_context(|| {
                format!("failed to load the tokenizer in {}", model_dir.display())
            })?,
        );
        let pad_token_id = resolve_pad_token_id(model_dir, &tokenizer);
        Self::from_parts(
            model,
            &args,
            tokenizer,
            pad_token_id,
            resolve_max_length(model_dir, max_length_override),
            batch_size,
        )
    }

    /// Assemble a reranker from an already-loaded backbone and tokenizer.
    ///
    /// Split from [`Self::load`] so the prompt assembly and the yes/no
    /// resolution can be exercised on a synthetic backbone.
    pub(crate) fn from_parts(
        model: Qwen3Model,
        args: &ModelArgs,
        tokenizer: MlxcelTokenizer,
        pad_token_id: u32,
        max_length: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let yes_id = single_token_id(&tokenizer, "yes")?;
        let no_id = single_token_id(&tokenizer, "no")?;
        for (word, id) in [("yes", yes_id), ("no", no_id)] {
            if args.vocab_size > 0 && id as usize >= args.vocab_size {
                bail!(
                    "the reranker answer token `{word}` is id {id}, outside the model's \
                     vocab_size {}",
                    args.vocab_size
                );
            }
        }
        let prefix_ids = tokenizer
            .encode(PROMPT_PREFIX, false)
            .context("failed to encode the reranker prompt prefix")?;
        let suffix_ids = tokenizer
            .encode(PROMPT_SUFFIX, false)
            .context("failed to encode the reranker prompt suffix")?;
        if prefix_ids.len() + suffix_ids.len() >= max_length {
            bail!(
                "the reranker prompt scaffold is {} tokens, which leaves no room inside the \
                 {max_length}-token limit",
                prefix_ids.len() + suffix_ids.len()
            );
        }
        tracing::info!(
            target: "mlxcel::rerank",
            yes_id,
            no_id,
            prefix_tokens = prefix_ids.len(),
            suffix_tokens = suffix_ids.len(),
            max_length,
            pad_token_id,
            batch_size,
            "Qwen3 generative reranker loaded"
        );
        Ok(Self {
            model,
            tokenizer,
            yes_id,
            no_id,
            prefix_ids,
            suffix_ids,
            pad_token_id,
            max_length,
            batch_size: batch_size.max(1),
        })
    }

    /// Tokens the pair itself may occupy once the scaffold is reserved.
    fn content_budget(&self) -> usize {
        self.max_length
            .saturating_sub(self.prefix_ids.len() + self.suffix_ids.len())
    }

    /// Build one prompt row: prefix, truncated content, suffix.
    pub(crate) fn encode_row(
        &self,
        instruction: &str,
        query: &str,
        document: &str,
    ) -> Result<Vec<u32>> {
        let content = prompt_content(instruction, query, document);
        let content_ids = self
            .tokenizer
            .encode(&content, false)
            .context("failed to encode a reranker query/document pair")?;
        let content_ids = truncate_token_ids(&content_ids, self.content_budget());
        let mut ids =
            Vec::with_capacity(self.prefix_ids.len() + content_ids.len() + self.suffix_ids.len());
        ids.extend_from_slice(&self.prefix_ids);
        ids.extend_from_slice(&content_ids);
        ids.extend_from_slice(&self.suffix_ids);
        Ok(ids)
    }

    /// One backbone pass over the `[B, L]` block `batch`: `[B, L, hidden]`
    /// states. Every pass of this reranker goes through here, which is what
    /// makes `forwarded` the count of token positions the backbone computed.
    fn forward_block(
        &self,
        batch: &EncodedBatch,
        caches: &mut [KVCache],
        mask: &MlxArray,
        forwarded: &mut usize,
    ) -> UniquePtr<MlxArray> {
        *forwarded += batch.batch * batch.width;
        self.model
            .forward_hidden(&batch.input_ids_array(), None, caches, Some(mask))
    }

    /// Score one micro-batch of whole prompt rows.
    fn score_rows(&self, rows: &[EncodedRow], forwarded: &mut usize) -> Result<Vec<f32>> {
        let batch =
            EncodedBatch::from_rows_with_padding(rows, self.pad_token_id, None, PaddingSide::Left);
        let mask = create_causal_padding_mask(&batch.attention_mask_array(), 0);
        let mut caches = self.model.make_caches();
        let hidden = self.forward_block(&batch, &mut caches, &mask, forwarded);
        // Left padding puts every row's last real token in the last column.
        let width = batch.width as i32;
        self.yes_probabilities(&slice_axis(&hidden, 1, width - 1, width))
    }

    /// Forward the run every row of a request opens with, alone and once.
    fn prefill_shared_run(&self, ids: &[u32], forwarded: &mut usize) -> Result<SharedRun> {
        let row = EncodedRow {
            ids: ids.to_vec(),
            type_ids: None,
        };
        let batch = EncodedBatch::from_rows(std::slice::from_ref(&row), self.pad_token_id, None);
        let mask = create_causal_padding_mask(&batch.attention_mask_array(), 0);
        let mut caches = self.model.make_caches();
        // Only the cache writes are wanted: the returned states are dropped
        // without being evaluated, and only the keys and values below are.
        let _ = self.forward_block(&batch, &mut caches, &mask, forwarded);
        let layers = caches
            .iter()
            .map(|cache| {
                let (keys, values) = cache
                    .visible_state()
                    .context("the shared prompt run left a layer without keys and values")?;
                // Evaluated here, so no micro-batch computes the run again.
                for array in [&keys, &values] {
                    mlxcel_core::try_eval(array).map_err(|e| {
                        anyhow::anyhow!("reranker shared-run evaluation failed: {e}")
                    })?;
                }
                Ok((keys, values))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SharedRun {
            len: ids.len(),
            layers,
        })
    }

    /// Score one micro-batch from the shared run: forward only what each row
    /// holds after it.
    fn score_tails(
        &self,
        shared: &SharedRun,
        rows: &[EncodedRow],
        forwarded: &mut usize,
    ) -> Result<Vec<f32>> {
        let tails: Vec<EncodedRow> = rows
            .iter()
            .map(|row| EncodedRow {
                ids: row.ids[shared.len..].to_vec(),
                type_ids: None,
            })
            .collect();
        // Right padding: a tail continues right after the run, so the padding
        // goes behind the tail, where the causal mask hides it from every
        // real token.
        let batch = EncodedBatch::from_rows(&tails, self.pad_token_id, None);
        let rows_in_batch = batch.batch as i32;
        let total = shared.len + batch.width;
        // The key axis is the run, real for every row, then the tail columns.
        let key_is_real: Vec<i32> = batch
            .token_counts
            .iter()
            .flat_map(|&count| (0..total).map(move |column| i32::from(column < shared.len + count)))
            .collect();
        let mask = create_causal_padding_mask(
            &mlxcel_core::from_slice_i32(&key_is_real, &[rows_in_batch, total as i32]),
            shared.len as i32,
        );

        let mut caches = self.model.make_caches();
        for (cache, (keys, values)) in caches.iter_mut().zip(&shared.layers) {
            cache.keys = Some(run_with_room(keys, rows_in_batch, batch.width as i32));
            cache.values = Some(run_with_room(values, rows_in_batch, batch.width as i32));
            // The next write lands at the run's end and is rotated from there.
            cache.offset = shared.len as i32;
        }
        let hidden = self.forward_block(&batch, &mut caches, &mask, forwarded);

        // `[1, 1, hidden]` per row, at the row's own last real token.
        let last: Vec<UniquePtr<MlxArray>> = batch
            .token_counts
            .iter()
            .enumerate()
            .map(|(row, &count)| {
                let (row, end) = (row as i32, count as i32);
                mlxcel_core::slice(&hidden, &[row, end - 1, 0], &[row + 1, end, i32::MAX])
            })
            .collect();
        let last: Vec<&MlxArray> = last.iter().map(|states| &**states).collect();
        self.yes_probabilities(&mlxcel_core::concatenate_many(&last, 0))
    }

    /// The probability of "yes" for each row of `[B, 1, hidden]` states read
    /// at the scoring position.
    fn yes_probabilities(&self, last: &MlxArray) -> Result<Vec<f32>> {
        // The head runs on one position per row instead of the whole
        // sequence: the `[B, L, vocab]` tensor would be hundreds of megabytes
        // for a long batch and every position but one is discarded.
        let logits = match &self.model.lm_head {
            Some(head) => head.forward(last),
            None => self.model.embed_tokens.as_linear(last),
        };
        let rows = mlxcel_core::array_shape(last)[0] as usize;
        let indices = mlxcel_core::from_slice_i32(&[self.yes_id as i32, self.no_id as i32], &[2]);
        // `[B, 1, vocab]` gathered on the vocabulary axis -> `[B, 1, 2]`.
        let picked = mlxcel_core::take(&logits, &indices, 2);
        let yes = slice_axis(&picked, 2, 0, 1);
        let no = slice_axis(&picked, 2, 1, 2);
        let scores = sigmoid_to_vec(&mlxcel_core::subtract(&yes, &no))?;
        if scores.len() != rows {
            bail!(
                "the Qwen3 reranker produced {} scores for {rows} documents",
                scores.len()
            );
        }
        Ok(scores)
    }

    /// Score encoded prompt rows in length-sorted micro-batches, stopping at
    /// the next micro-batch boundary once `is_abandoned()` returns `true`.
    ///
    /// The first `shared_len` tokens, a run [`shared_run_len`] found in these
    /// rows, are forwarded once and each micro-batch forwards only its rows'
    /// tails. `0` forwards every row whole.
    ///
    /// Returns the scores in row order with the tokens of every full row, and
    /// the token positions the backbone computed.
    fn score_encoded(
        &self,
        rows: &[EncodedRow],
        shared_len: usize,
        is_abandoned: &dyn Fn() -> bool,
    ) -> Result<(RerankScores, usize)> {
        let lengths: Vec<usize> = rows.iter().map(|row| row.ids.len()).collect();
        let mut scores = vec![0.0f32; rows.len()];
        let mut prompt_tokens = 0usize;
        let mut forwarded = 0usize;
        let mut shared: Option<SharedRun> = None;
        for chunk in batching_order(&lengths).chunks(self.batch_size) {
            if is_abandoned() {
                bail!(super::RerankAbandoned);
            }
            let chunk_rows: Vec<EncodedRow> =
                chunk.iter().map(|&index| rows[index].clone()).collect();
            // Either scorer returns exactly one score per row, so every index
            // of the chunk is written.
            let chunk_scores = if shared_len == 0 {
                self.score_rows(&chunk_rows, &mut forwarded)?
            } else {
                // Prefilled by the first micro-batch, after its abandon check.
                let run = match &shared {
                    Some(run) => run,
                    None => shared.insert(
                        self.prefill_shared_run(&rows[0].ids[..shared_len], &mut forwarded)?,
                    ),
                };
                self.score_tails(run, &chunk_rows, &mut forwarded)?
            };
            prompt_tokens += chunk.iter().map(|&index| lengths[index]).sum::<usize>();
            for (&index, score) in chunk.iter().zip(chunk_scores) {
                scores[index] = score;
            }
        }
        Ok((
            RerankScores {
                scores,
                prompt_tokens,
            },
            forwarded,
        ))
    }
}

/// The token run every row of a request opens with, forwarded once: each
/// layer's `[1, H, len, D]` keys and values.
struct SharedRun {
    len: usize,
    layers: Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

/// Fewest tokens a pass of the shared-run path carries. A one-token step
/// takes the attention block's single-token decode kernels, which no
/// reranker pass has ever run.
const MIN_PASS_TOKENS: usize = 2;

/// Tokens all `rows` open with, the run a request forwards once.
///
/// It is the longest common prefix of the token rows, shortened so the
/// shortest row keeps [`MIN_PASS_TOKENS`] of its own. `0` means there is
/// nothing to share: one row, or a common prefix below [`MIN_PASS_TOKENS`].
fn shared_run_len(rows: &[EncodedRow]) -> usize {
    let Some((first, rest)) = rows.split_first() else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let common = rest.iter().fold(first.ids.len(), |len, row| {
        first.ids[..len]
            .iter()
            .zip(&row.ids)
            .take_while(|(a, b)| a == b)
            .count()
    });
    let shortest = rows.iter().map(|row| row.ids.len()).min().unwrap_or(0);
    let shared = common.min(shortest.saturating_sub(MIN_PASS_TOKENS));
    if shared < MIN_PASS_TOKENS { 0 } else { shared }
}

/// The cache buffer a micro-batch starts from: the shared run's
/// `[1, H, P, D]` keys or values repeated for `rows` rows, then `room`
/// unwritten slots for the tails, `[rows, H, P + room, D]`.
fn run_with_room(shared: &MlxArray, rows: i32, room: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(shared);
    let run = mlxcel_core::broadcast_to(shared, &[rows, shape[1], shape[2], shape[3]]);
    let unwritten = mlxcel_core::zeros(
        &[rows, shape[1], room, shape[3]],
        mlxcel_core::array_dtype(shared),
    );
    mlxcel_core::concatenate(&run, &unwritten, 2)
}

/// Row indices in the order the micro-batches take them: longest row first,
/// rows of equal length in arrival order.
///
/// A micro-batch is padded to its longest row, so cutting this order into
/// batches keeps rows of similar length together. The sort is stable, which
/// makes the batches a function of the row lengths alone.
fn batching_order(lengths: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(lengths[index]));
    order
}

/// `max_length` for a generative reranker: the shared 8192 ceiling, lowered by
/// the checkpoint's declared limits and by `--rerank-max-length`.
///
/// The position table is RoPE, so `max_position_embeddings` (40960 on the 0.6B
/// checkpoint, 262144 on the VL one) is not a cap worth reading; the ceiling is
/// what keeps one pair's prefill bounded.
pub(crate) fn resolve_max_length(model_dir: &Path, override_value: Option<usize>) -> usize {
    derive_max_length(model_dir, false, override_value).min(RERANK_MAX_LENGTH_CAP)
}

impl Reranker for Qwen3Reranker {
    fn kind(&self) -> RerankerKind {
        RerankerKind::GenerativeText
    }

    fn score_until(
        &self,
        query: &RerankItem,
        documents: &[RerankItem],
        instruction: Option<&str>,
        is_abandoned: &dyn Fn() -> bool,
    ) -> Result<RerankScores> {
        if query.has_image() || documents.iter().any(RerankItem::has_image) {
            bail!("the Qwen3 reranker is text-only; use a Qwen3-VL reranker for image documents");
        }
        let instruction = resolve_instruction(instruction);
        let query_text = query.text_or_empty();
        let rows = documents
            .iter()
            .map(|document| {
                Ok(EncodedRow {
                    ids: self.encode_row(instruction, query_text, document.text_or_empty())?,
                    type_ids: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (scores, _) = self.score_encoded(&rows, shared_run_len(&rows), is_abandoned)?;
        Ok(scores)
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
#[path = "qwen3_generative_tests.rs"]
mod qwen3_generative_tests;
