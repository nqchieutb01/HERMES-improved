"""Batched answer decoding over independent HERMES memories (Qwen3-VL).

Answering dominates inference time and is bound by reading the weights once per generated token, so decoding several
questions together is close to free per extra question. Each question keeps its own memory: a KV cache whose length
differs per layer (HERMES prunes layer-wise) and its own M-RoPE positions. The memories are packed into right-padded
per-layer buffers and decoded in lockstep with flash-attention's KV-cache kernel, which takes a separate valid length
per sequence, so no token attends to padding.

The prompt prefill and the first token use the unbatched code path (``prefill_job``); the token selection rules are the
same as ``QwenVL_Hermes.question_answering`` (greedy with logit repetition penalty) and
``Qwen3VL_Hermes.contrastive_answering`` (contrast against a counterfactual memory). Only the floating-point order of
the batched matrix products differs from batch size 1.
"""

from __future__ import annotations

import math

import torch
from flash_attn import flash_attn_with_kvcache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb


def _layer_kv(cache, layer_idx):
    k, v = cache[layer_idx]
    return k, v


@torch.inference_mode()
def prefill_job(model, prompt, negative_state=None, clone=False, contrastive=None):
    """Read the prompt into the current memory (and the counterfactual one) and return a decoding job.

    The model's memory is left as before the call (prompt tokens removed), so encoding can continue. With ``clone``
    the job keeps its own copy of the caches (needed when the live memory is later modified in place, as in streaming).
    """
    tok = model.processor.tokenizer
    ids = torch.as_tensor([tok(prompt).input_ids], device=model.device)
    model._ensure_dynamic_cache()
    before = model._get_cache_seq_len_per_layer()
    real = model.memory_snapshot()

    def row():
        logits = _prefill_logits(model, ids)
        kv = [_layer_kv(model.kv_cache, i) for i in range(model.num_layers)]
        if clone:
            kv = [(k.clone(), v.clone()) for k, v in kv]
        return {"kv": kv, "offsets": model._get_next_global_offset_per_layer(), "logits": logits}

    rows = [row()]
    if negative_state is not None:
        model.memory_restore(negative_state)
        model._ensure_dynamic_cache()
        rows.append(row())
        model.memory_restore(real)
    model._truncate_kv_cache(before)
    for layer_idx in range(model.num_layers):
        pos = model._position_ids_cache[layer_idx]
        if pos is not None and pos.shape[1] > before[layer_idx]:
            model._position_ids_cache[layer_idx] = pos[:, :before[layer_idx]].contiguous()
    return {"rows": rows, "contrastive": contrastive}


def _prefill_logits(model, ids):
    """``Qwen3VL_Hermes._forward_text`` without the float cast (greedy answering works on the raw logits)."""
    offsets = model._get_next_global_offset_per_layer()
    q_len = ids.shape[1]
    model._layer_position_ids.clear()
    for layer_idx in range(model.num_layers):
        model._layer_position_ids[layer_idx] = model._build_position_ids_3d_for_text(offsets[layer_idx], q_len, 1)
    out = model.language_model(inputs_embeds=model.get_input_embeddings()(ids), use_cache=True,
                               past_key_values=model.kv_cache,
                               position_ids=model._build_position_ids_3d_for_text(offsets[0], q_len, 1))
    model.kv_cache = out.past_key_values
    for layer_idx in range(model.num_layers):
        o = offsets[layer_idx]
        model._append_position_ids_layer(layer_idx, [o, o, o], q_len)
    model._layer_position_ids.clear()
    return model.lm_head(out.last_hidden_state)[0, -1]


class _PackedGroup:
    """Right-padded per-layer KV buffers [rows, capacity, kv_heads, head_dim] with a valid length per row and layer."""

    def __init__(self, rows, num_layers, extra):
        self.k, self.v, self.lens = [], [], []
        for layer_idx in range(num_layers):
            lengths = [r["kv"][layer_idx][0].shape[2] for r in rows]
            k0 = rows[0]["kv"][layer_idx][0]
            shape = (len(rows), max(lengths) + extra, k0.shape[1], k0.shape[3])
            k_buf = torch.zeros(shape, dtype=k0.dtype, device=k0.device)
            v_buf = torch.zeros(shape, dtype=k0.dtype, device=k0.device)
            for i, r in enumerate(rows):
                k, v = r["kv"][layer_idx]
                k_buf[i, :lengths[i]] = k[0].transpose(0, 1)
                v_buf[i, :lengths[i]] = v[0].transpose(0, 1)
                r["kv"][layer_idx] = None  # release the job's own copy as soon as it is packed
            self.k.append(k_buf)
            self.v.append(v_buf)
            self.lens.append(torch.tensor(lengths, dtype=torch.int32, device=k0.device))
        self.offsets = torch.tensor([r["offsets"] for r in rows], dtype=torch.float32, device=k0.device)

    def keep(self, index):
        index = torch.as_tensor(index, device=self.offsets.device)
        self.k = [x.index_select(0, index) for x in self.k]
        self.v = [x.index_select(0, index) for x in self.v]
        self.lens = [x.index_select(0, index) for x in self.lens]
        self.offsets = self.offsets.index_select(0, index)


class _PackedMemory:
    """Groups of rows decoded together; each group is padded separately (real memories are long, the no-video
    counterfactual is only the prompt), and the linear layers run on all rows at once."""

    def __init__(self, groups, num_layers, extra):
        self.groups = [_PackedGroup(rows, num_layers, extra) for rows in groups]

    @property
    def offsets(self):
        return torch.cat([g.offsets for g in self.groups])

    def keep(self, index):
        for g in self.groups:
            g.keep(index)


@torch.inference_mode()
def _decode_step(model, memory, token_ids, step):
    """One token for every row: the Qwen3-VL text decoder with per-row, per-layer positions and cache lengths."""
    lm = model.language_model
    rows, num_layers = token_ids.shape[0], len(lm.layers)
    hidden = lm.embed_tokens(token_ids.view(rows, 1))
    # Text tokens use the same position on all three M-RoPE axes: offset of the row in that layer + step.
    pos = (memory.offsets.t() + step).reshape(1, num_layers * rows, 1).expand(3, -1, -1)
    cos, sin = lm.rotary_emb(hidden, pos)
    cos, sin = cos.view(num_layers, rows, 1, -1), sin.view(num_layers, rows, 1, -1)
    for layer_idx, layer in enumerate(lm.layers):
        attn = layer.self_attn
        residual = hidden
        h = layer.input_layernorm(hidden)
        shape = (rows, 1, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(h).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
        v = attn.v_proj(h).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos[layer_idx], sin[layer_idx])
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        outs, start = [], 0
        for g in memory.groups:
            stop = start + g.offsets.shape[0]
            outs.append(flash_attn_with_kvcache(q[start:stop], g.k[layer_idx], g.v[layer_idx],
                                                k=k[start:stop], v=v[start:stop], cache_seqlens=g.lens[layer_idx],
                                                softmax_scale=attn.scaling, causal=True))
            g.lens[layer_idx] += 1
            start = stop
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        hidden = residual + attn.o_proj(out.reshape(rows, 1, -1))
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
    return model.lm_head(lm.norm(hidden))[:, -1]


@torch.inference_mode()
def batched_answering(model, jobs, max_new_tokens=384, repetition_penalty=1.1, alpha=1.0, beta=0.1, scope="all",
                      adaptive=False, rule="pmi", token_log=None):
    """Decode all jobs together; returns the answer text of each job (same order).

    Jobs without a counterfactual row decode greedily (``question_answering``); jobs with one are contrasted against
    it (``contrastive_answering``). All jobs of a call must be of the same kind. ``token_log`` (diagnostic): a list
    that receives, per job, the generated token ids and the margin between the best and second-best score per step.
    """
    tok = model.processor.tokenizer
    eos = tok.eos_token_id
    contrastive = len(jobs[0]["rows"]) == 2
    assert all((len(j["rows"]) == 2) == contrastive for j in jobs), "mixed greedy and contrastive jobs"
    n = len(jobs)
    # Rows: the real memory of every job, then (contrastive) every counterfactual memory.
    groups = [[j["rows"][0] for j in jobs]] + ([[j["rows"][1] for j in jobs]] if contrastive else [])
    logits = torch.stack([r["logits"] for g in groups for r in g])
    memory = _PackedMemory(groups, model.num_layers, max_new_tokens + 1)
    del groups
    for j in jobs:
        j["rows"] = None
    vocab = logits.shape[-1]
    seen = torch.zeros((n, vocab), dtype=torch.bool, device=logits.device)
    out_ids = [[] for _ in range(n)]
    margins = [[] for _ in range(n)]
    active = list(range(n))  # job ids still decoding, in row order
    log_beta, log_penalty = math.log(beta), math.log(repetition_penalty)
    for step in range(max_new_tokens):
        m = len(active)
        if contrastive:
            lp = torch.log_softmax(logits[:m].float(), -1)
            ln = torch.log_softmax(logits[m:].float(), -1)
            top = lp.max(-1, keepdim=True).values
            on = [scope == "all" or model._in_time_slot(tok.decode(out_ids[j]), scope) for j in active]
            if adaptive:
                conf = top.exp().view(-1).tolist()
                a = [alpha * (1.0 - c) for c in conf]
            else:
                a = [alpha] * m
            a_t = torch.tensor(a, dtype=torch.float32, device=lp.device).view(-1, 1)
            if rule == "against":
                contrasted = lp - a_t * torch.clamp(ln - lp, min=0.0)
            else:
                contrasted = (1.0 + torch.tensor(a, dtype=torch.float64).view(-1, 1)).to(lp.device, torch.float32) * lp \
                    - a_t * ln
            contrasted = contrasted.masked_fill(lp < top + log_beta, float("-inf"))
            on_t = torch.tensor(on, device=lp.device).view(-1, 1)
            score = torch.where(on_t, contrasted, lp)
            score = torch.where(seen[active], score - log_penalty, score)
            tokens = torch.argmax(score, -1).tolist()
            if token_log is not None:
                best2 = torch.topk(score, 2, dim=-1).values
                for j, (b1, b2) in zip(active, best2.tolist()):
                    margins[j].append(b1 - b2)
        else:
            last = logits[:m]
            if repetition_penalty != 1.0:
                penalised = torch.where(last < 0, last * repetition_penalty, last / repetition_penalty)
                last = torch.where(seen[active], penalised, last)
            tokens = torch.topk(last, 1, dim=-1).indices.view(-1).tolist()
            if token_log is not None:
                best2 = torch.topk(last.float(), 2, dim=-1).values
                for j, (b1, b2) in zip(active, best2.tolist()):
                    margins[j].append(b1 - b2)
        finished = []
        for row, (j, t) in enumerate(zip(active, tokens)):
            out_ids[j].append(t)
            if t == eos:
                finished.append(row)
        seen[torch.tensor(active, device=seen.device), torch.tensor(tokens, device=seen.device)] = True
        if len(finished) == m:
            break
        if step == max_new_tokens - 1:
            break
        if finished:
            keep = [r for r in range(m) if r not in finished]
            active = [active[r] for r in keep]
            tokens = [tokens[r] for r in keep]
            memory.keep(keep)
        ids = torch.tensor(tokens * (2 if contrastive else 1), device=logits.device)
        logits = _decode_step(model, memory, ids, step)
    del memory
    torch.cuda.empty_cache()
    if token_log is not None:
        token_log.extend({"tokens": o, "margins": [round(x, 4) for x in mg]} for o, mg in zip(out_ids, margins))
    return [tok.decode(o, skip_special_tokens=True, spaces_between_special_tokens=False,
                       clean_up_tokenization_spaces=True) for o in out_ids]
