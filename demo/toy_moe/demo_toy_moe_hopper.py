#!/usr/bin/env python3
"""Tiny MoE model for reasoning about resident/streaming task graph mechanics.

This script builds a 2-layer MoE transformer with random weights
(no HuggingFace) that are intentionally constructed to produce the same kind
of irregular, asymmetric execution patterns as a real MoE model.

Sources of irregularity (matching real workloads):

1. MoE routing skew: Gate weights use Zipf-like scaling so expert 0 is
   ~80x more likely to be selected than expert 7.  With 4 tokens and
   topk=2, expect something like: expert 0 gets 4 tokens, expert 1 gets
   3, expert 3 gets 1, experts 5-7 get 0.  W13/W2 tasks for hot experts
   do real work; cold expert tasks are near-zero-cost no-ops.  The task
   graph cannot know this — all W13 tasks look identical in the DAG.

2. Variable attention work: 2 requests with different prompt lengths
   (request 0: length 3, request 1: length 1).  Paged attention tasks
   for request 0 do 3x the memory traffic of request 1.

3. Batch-partitioned tasks: With B=4 tokens, RMSNorm, SILU_MUL, and
   MUL_SUM_ADD get 4 tasks each.  Each token's data path through the
   MoE section is independent — token 0 might route to experts {0,1}
   while token 2 routes to {0,3}, creating different dependency fan-out
   per data item.

4. Layer-to-layer routing drift: Each layer has independently biased
   gate weights (different random seed per layer, same Zipf envelope).
   Layer 0 might route heavily to expert 0, layer 1 to expert 1.  This
   means the "hot" expert changes across layers, which streaming mode
   must handle.

Usage:
    # Legacy event mode (baseline):
    MIRAGE_TASK_GRAPH_MODE=legacy_event \\
    python3 demo/toy_moe/demo_toy_moe_hopper.py --profiling --output-dir out/legacy

    # Resident hybrid prelaunch:
    MIRAGE_TASK_GRAPH_MODE=resident_data \\
    MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch \\
    python3 demo/toy_moe/demo_toy_moe_hopper.py --profiling --output-dir out/resident

    # Streaming legacy base:
    MIRAGE_TASK_GRAPH_MODE=streaming_data \\
    python3 demo/toy_moe/demo_toy_moe_hopper.py --profiling --output-dir out/stream_legacy

    # Streaming hybrid base:
    MIRAGE_TASK_GRAPH_MODE=streaming_data \\
    MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch \\
    python3 demo/toy_moe/demo_toy_moe_hopper.py --profiling --output-dir out/stream_hybrid

Model dimensions (all chosen to minimise task count while preserving
the structure of a real MoE transformer):
    hidden_size        = 512
    head_dim           = 128
    num_q_heads        = 4   (512 / 128)
    num_kv_heads       = 4
    num_experts        = 128
    num_experts_per_tok = 2
    intermediate_size  = 128  (MoE FFN hidden)
    vocab_size         = 256
    num_layers         = 2

Default batch: 4 tokens, 2 requests (prompt lengths 3 and 1).

The toy uses the smallest shapes that satisfy the Hopper kernels used by this
demo path:
    hidden_size >= 512 for the bfloat16 RMSNorm implementation
    num_experts in {128, 256} for the fused top-k routing implementation
"""

import argparse
import os
import torch

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 2
HIDDEN_SIZE = 512
HEAD_DIM = 128
NUM_Q_HEADS = HIDDEN_SIZE // HEAD_DIM       # 4
NUM_KV_HEADS = NUM_Q_HEADS                  # 4 (MHA)
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 2
INTERMEDIATE_SIZE = 128                      # per-expert FFN hidden
VOCAB_SIZE = 256                             # padded, divisible by 128
ROPE_THETA = 10000.0

# Fused QKV output dim: (num_q + 2*num_kv) * head_dim
FUSED_QKV_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM  # 768

# ---------------------------------------------------------------------------
# Expert routing skew: Zipf-like gate-weight scaling.
#
# Each expert i gets its gate row scaled by scale[i] = base_scale * decay^i.
# The actual routing still depends on the (random) hidden-state input, so it
# is not deterministic, but the statistical skew is strong enough that a small
# hot subset of experts dominates most tokens.
# ---------------------------------------------------------------------------
EXPERT_GATE_DECAY = 0.55   # per-rank multiplicative decay
EXPERT_GATE_BASE = 4.0     # scale for expert 0


def make_skewed_gate_weights(num_experts: int, hidden_size: int,
                             seed_offset: int = 0) -> torch.Tensor:
    """Create gate weight matrix with Zipf-like expert preference.

    Returns shape (num_experts, hidden_size) on CUDA.
    Each expert row is an independent random direction, but scaled so
    expert 0 has much larger norm than expert 7.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(42 + seed_offset)
    raw = torch.randn(num_experts, hidden_size, device="cuda", generator=gen)
    scales = torch.tensor(
        [EXPERT_GATE_BASE * (EXPERT_GATE_DECAY ** i) for i in range(num_experts)],
        device="cuda", dtype=torch.bfloat16,
    )
    return raw * scales.unsqueeze(1)


def make_rope_embeddings(max_seq_len: int, head_dim: int,
                         theta: float = 10000.0):
    """Generate cos/sin RoPE position embeddings on CUDA."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                              dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_emb = emb.cos().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    sin_emb = emb.sin().unsqueeze(0).to(dtype=torch.bfloat16, device="cuda")
    return cos_emb, sin_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy MoE Hopper demo")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trace-name", type=str, default="")
    parser.add_argument("--max-seq-length", type=int, default=8,
                        help="Total sequence length (prompt + generated)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4,
                        help="Tokens in flight per iteration (B)")
    parser.add_argument("--max-num-batched-requests", type=int, default=2,
                        help="Concurrent requests with different prompt lengths")
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-num-pages", type=int, default=4)
    args = parser.parse_args()

    print("Toy MoE demo args:", args)

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)

    import mirage as mi

    B = args.max_num_batched_tokens
    R = args.max_num_batched_requests

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------
    # Use a fixed seed for reproducibility, but the routing will still
    # vary with batch content because gate(x) = W_gate @ x.
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    embed_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device="cuda")

    layer_weights = []
    moe_gate_up_weights = []
    moe_down_weights = []
    for layer_idx in range(NUM_LAYERS):
        lw = {
            "input_layernorm": torch.randn(HIDDEN_SIZE, device="cuda"),
            "qkv_proj": torch.randn(FUSED_QKV_DIM, HIDDEN_SIZE, device="cuda"),
            "q_norm": torch.randn(HEAD_DIM, device="cuda"),
            "k_norm": torch.randn(HEAD_DIM, device="cuda"),
            "o_proj": torch.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM,
                                  device="cuda"),
            "post_attn_layernorm": torch.randn(HIDDEN_SIZE, device="cuda"),
            # Skewed gate: each layer gets a different random seed so the
            # "hot" expert drifts across layers.
            "moe_gate": make_skewed_gate_weights(
                NUM_EXPERTS, HIDDEN_SIZE, seed_offset=layer_idx * 1000),
        }
        layer_weights.append(lw)

        gate_up = torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
                              device="cuda")
        moe_gate_up_weights.append(gate_up)
        down = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                           device="cuda")
        moe_down_weights.append(down)

    final_norm_weight = torch.randn(HIDDEN_SIZE, device="cuda")
    lm_head_weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, device="cuda")

    # KV cache
    key_cache = torch.zeros(
        NUM_LAYERS, args.max_num_pages, args.page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=torch.bfloat16, device="cuda",
    )
    value_cache = torch.zeros(
        NUM_LAYERS, args.max_num_pages, args.page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=torch.bfloat16, device="cuda",
    )

    # RoPE
    cos_emb, sin_emb = make_rope_embeddings(args.page_size, HEAD_DIM,
                                            ROPE_THETA)

    # ------------------------------------------------------------------
    # Meta tensors — asymmetric request shapes
    #
    # With R=2 requests and max_seq_length=8:
    #   request 0: prompt_length = 3  (longer context, more attention work)
    #   request 1: prompt_length = 1  (short, less attention work)
    #
    # This creates variable-duration paged-attention tasks, which is the
    # main source of non-MoE irregularity in a real serving workload.
    # ------------------------------------------------------------------
    tokens = torch.zeros(R, args.max_seq_length, dtype=torch.long,
                         device="cuda")
    # Fill prompts with distinct token ids so the hidden states diverge
    # and each token routes to different experts.
    prompt_lens = []
    for r in range(R):
        plen = min(3 if r == 0 else 1, args.max_seq_length - 1)
        prompt_lens.append(plen)
        for p in range(plen):
            tokens[r, p] = (r * 50 + p + 1) % VOCAB_SIZE
    prompt_lengths = torch.tensor(prompt_lens, dtype=torch.int, device="cuda")

    input_tokens = torch.zeros(B, 1, dtype=torch.long, device="cuda")
    # Seed input tokens with different values so each token's hidden
    # state is different, leading to different gate activations.
    for t in range(B):
        input_tokens[t, 0] = (t * 37 + 7) % VOCAB_SIZE

    output_tokens = torch.zeros(B, 1, dtype=torch.long, device="cuda")
    step = torch.zeros(R, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.ones(R, dtype=torch.int32, device="cuda")

    qo_indptr_buffer = torch.empty(R + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(R + 1, dtype=torch.int32,
                                         device="cuda")
    paged_kv_indices_buffer = torch.empty(args.max_num_pages, dtype=torch.int32,
                                          device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(R, dtype=torch.int32,
                                                device="cuda")

    # Profiler buffer
    if args.profiling:
        profiler_tensor = torch.zeros(
            max(5000 * 128,
                256 * args.max_seq_length * B * R * 128),
            dtype=torch.uint64, device="cuda",
        ).contiguous()
    else:
        profiler_tensor = None

    # ------------------------------------------------------------------
    # Build PersistentKernel graph
    # ------------------------------------------------------------------
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=R,
        max_num_batched_tokens=B,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=-1,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=None,
        use_cutlass_kernel=True,
    )

    # ---- Attach inputs and allocate intermediate tensors ----
    x_input = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    cos_pe = mpk.attach_input(
        torch_tensor=cos_emb[0, :args.page_size, :],
        name="cos_position_embedding")
    sin_pe = mpk.attach_input(
        torch_tensor=sin_emb[0, :args.page_size, :],
        name="sin_position_embedding")

    y = mpk.new_tensor(dims=(B, HIDDEN_SIZE), dtype=mi.bfloat16,
                       name="embed_out", io_category="cuda_tensor")
    rmsnorm_out_qkv = mpk.new_tensor(dims=(B, HIDDEN_SIZE), dtype=mi.bfloat16,
                                     name="rmsnorm_out_qkv",
                                     io_category="cuda_tensor")
    attn_in = mpk.new_tensor(dims=(B, FUSED_QKV_DIM), dtype=mi.bfloat16,
                             name="attn_in", io_category="cuda_tensor")
    attn_out = mpk.new_tensor(dims=(B, NUM_Q_HEADS * HEAD_DIM),
                              dtype=mi.bfloat16, name="attn_out",
                              io_category="cuda_tensor")
    attn_proj_out = mpk.new_tensor(dims=(B, HIDDEN_SIZE), dtype=mi.bfloat16,
                                   name="attn_proj_out",
                                   io_category="cuda_tensor")
    rmsnorm_out_moe = mpk.new_tensor(dims=(B, HIDDEN_SIZE), dtype=mi.bfloat16,
                                     name="rmsnorm_out_moe",
                                     io_category="cuda_tensor")
    moe_gate_out = mpk.new_tensor(dims=(B, NUM_EXPERTS), dtype=mi.bfloat16,
                                  name="moe_gate_out",
                                  io_category="cuda_tensor")
    moe_routing_indices = mpk.new_tensor(dims=(NUM_EXPERTS, B), dtype=mi.int32,
                                         name="moe_routing_indices",
                                         io_category="cuda_tensor")
    moe_mask = mpk.new_tensor(dims=(NUM_EXPERTS + 1,), dtype=mi.int32,
                              name="moe_mask", io_category="cuda_tensor")
    moe_topk_weight = mpk.new_tensor(dims=(B, NUM_EXPERTS_PER_TOK),
                                     dtype=mi.float32, name="moe_topk_weight",
                                     io_category="cuda_tensor")
    mlp_mid = mpk.new_tensor(
        dims=(B, NUM_EXPERTS_PER_TOK, 2 * INTERMEDIATE_SIZE),
        dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    silu_mul_out = mpk.new_tensor(
        dims=(B, NUM_EXPERTS_PER_TOK, INTERMEDIATE_SIZE),
        dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mlp_out = mpk.new_tensor(
        dims=(B, NUM_EXPERTS_PER_TOK, HIDDEN_SIZE),
        dtype=mi.bfloat16, name="mlp_out", io_category="cuda_tensor")
    mlp_weighted_sum_out = mpk.new_tensor(dims=(B, HIDDEN_SIZE),
                                          dtype=mi.bfloat16,
                                          name="mlp_weighted_sum_out",
                                          io_category="cuda_tensor")
    rmsnorm_out_final = mpk.new_tensor(dims=(B, HIDDEN_SIZE),
                                       dtype=mi.bfloat16,
                                       name="rmsnorm_out_final",
                                       io_category="cuda_tensor")
    argmax_in = mpk.new_tensor(dims=(B, VOCAB_SIZE), dtype=mi.bfloat16,
                               name="argmax_in", io_category="cuda_tensor")
    argmax_part_value = mpk.new_tensor(dims=(B, mpk.num_workers),
                                       dtype=mi.bfloat16,
                                       name="argmax_part_value",
                                       io_category="cuda_tensor")
    argmax_part_index = mpk.new_tensor(dims=(B, mpk.num_workers),
                                       dtype=mi.int64,
                                       name="argmax_part_index",
                                       io_category="cuda_tensor")
    argmax_out = mpk.attach_input(torch_tensor=output_tokens,
                                  name="output_token")

    # ---- Grid dim calculations ----
    # QKV linear: partition output dim
    qkv_grid = (min(FUSED_QKV_DIM // 128, 4) if FUSED_QKV_DIM % 128 == 0
                else FUSED_QKV_DIM // 64)
    # o_proj linear with residual: hidden_size / 64
    oproj_grid = HIDDEN_SIZE // 64                          # 4
    # MoE gate linear
    gate_grid = max(1, NUM_EXPERTS // 8)                    # 1
    # W13: (expert_groups, output_tiles)
    w13_grid_x = 2
    w13_grid_y = max(1, (2 * INTERMEDIATE_SIZE) // 128)     # 2
    # W2: (expert_groups, output_tiles)
    w2_grid_x = 2
    w2_grid_y = max(1, HIDDEN_SIZE // 128)                  # 2
    # SILU_MUL: (batch, experts_per_tok)
    silu_grid = (B, NUM_EXPERTS_PER_TOK, 1)
    # MUL_SUM_ADD: (batch, hidden/256)
    msa_grid = (B, max(1, HIDDEN_SIZE // 256), 1)
    # lm_head
    lm_head_grid = min(mpk.num_workers, max(1, VOCAB_SIZE // 128))  # 2
    # argmax
    argmax_partial_grid = (mpk.num_workers, 1, 1)

    # ---- Build graph: Embed ----
    w_embed = mpk.attach_input(torch_tensor=embed_weight, name="embed_tokens")
    mpk.embed_layer(
        input=x_input, weight=w_embed, output=y,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1), input_source=1,
    )
    x = y

    # ---- Build graph: Transformer layers ----
    for i in range(NUM_LAYERS):
        lw = layer_weights[i]

        # -- Pre-attention RMSNorm --
        w_norm = mpk.attach_input(torch_tensor=lw["input_layernorm"],
                                  name=f"layer_{i}_input_layernorm")
        mpk.rmsnorm_layer(
            input=x, weight=w_norm, output=rmsnorm_out_qkv,
            grid_dim=(B, 1, 1), block_dim=(256, 1, 1),
        )

        # -- QKV linear --
        w_qkv = mpk.attach_input(torch_tensor=lw["qkv_proj"],
                                 name=f"layer_{i}_qkv_proj")
        mpk.linear_layer(
            input=rmsnorm_out_qkv, weight=w_qkv, output=attn_in,
            grid_dim=(qkv_grid, 1, 1), block_dim=(256, 1, 1),
        )

        # -- Paged attention --
        # grid_dim = (num_requests, num_kv_heads, 1)
        # With R=2 and different prompt_lengths, the two request rows
        # of this grid have different amounts of work.
        w_q_norm = mpk.attach_input(torch_tensor=lw["q_norm"],
                                    name=f"layer_{i}_q_norm")
        w_k_norm = mpk.attach_input(torch_tensor=lw["k_norm"],
                                    name=f"layer_{i}_k_norm")
        k_cache = mpk.attach_input(torch_tensor=key_cache[i],
                                   name=f"layer_{i}_k_cache")
        v_cache = mpk.attach_input(torch_tensor=value_cache[i],
                                   name=f"layer_{i}_v_cache")
        mpk.paged_attention_layer(
            input=attn_in, k_cache=k_cache, v_cache=v_cache,
            q_norm=w_q_norm, k_norm=w_k_norm,
            cos_pos_embed=cos_pe, sin_pos_embed=sin_pe,
            output=attn_out,
            grid_dim=(R, NUM_KV_HEADS, 1),
            block_dim=(256, 1, 1),
        )

        # -- o_proj linear with residual --
        w_o = mpk.attach_input(torch_tensor=lw["o_proj"],
                               name=f"layer_{i}_o_proj")
        mpk.linear_with_residual_layer(
            input=attn_out, weight=w_o, residual=x, output=attn_proj_out,
            grid_dim=(oproj_grid, 1, 1), block_dim=(256, 1, 1),
        )
        x = attn_proj_out

        # -- Pre-MoE RMSNorm --
        w_post_norm = mpk.attach_input(
            torch_tensor=lw["post_attn_layernorm"],
            name=f"layer_{i}_post_attn_layernorm")
        mpk.rmsnorm_layer(
            input=x, weight=w_post_norm, output=rmsnorm_out_moe,
            grid_dim=(B, 1, 1), block_dim=(256, 1, 1),
        )

        # -- MoE gate linear --
        w_gate = mpk.attach_input(torch_tensor=lw["moe_gate"],
                                  name=f"layer_{i}_moe_gate")
        mpk.linear_layer(
            input=rmsnorm_out_moe, weight=w_gate, output=moe_gate_out,
            grid_dim=(gate_grid, 1, 1), block_dim=(256, 1, 1),
        )

        # -- TopK + softmax routing --
        mpk.moe_topk_softmax_routing_layer(
            input=moe_gate_out,
            output=(moe_topk_weight, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        )

        # -- MoE W13 linear (gate+up fused) --
        w_gatedup = mpk.attach_input(torch_tensor=moe_gate_up_weights[i],
                                     name=f"layer_{i}_gate_proj")
        mpk.moe_w13_linear_layer(
            input=rmsnorm_out_moe, weight=w_gatedup,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=mlp_mid,
            grid_dim=(w13_grid_x, w13_grid_y, 1), block_dim=(256, 1, 1),
        )

        # -- SiLU * mul --
        mpk.moe_silu_mul_layer(
            input=mlp_mid, output=silu_mul_out,
            grid_dim=silu_grid, block_dim=(256, 1, 1),
        )

        # -- MoE W2 linear (down proj) --
        w_down = mpk.attach_input(torch_tensor=moe_down_weights[i],
                                  name=f"layer_{i}_down_proj")
        mpk.moe_w2_linear_layer(
            input=silu_mul_out, weight=w_down,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=mlp_out,
            grid_dim=(w2_grid_x, w2_grid_y, 1), block_dim=(256, 1, 1),
        )

        # -- MoE weighted sum + residual add --
        mpk.moe_mul_sum_add_layer(
            input=mlp_out, weight=moe_topk_weight, residual=x,
            output=mlp_weighted_sum_out,
            grid_dim=msa_grid, block_dim=(256, 1, 1),
        )
        x = mlp_weighted_sum_out

    # ---- Final RMSNorm + lm_head ----
    w_final_norm = mpk.attach_input(torch_tensor=final_norm_weight,
                                    name="model_norm_weight")
    mpk.rmsnorm_layer(
        input=x, weight=w_final_norm, output=rmsnorm_out_final,
        grid_dim=(B, 1, 1), block_dim=(256, 1, 1),
    )
    w_lm_head = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.linear_layer(
        input=rmsnorm_out_final, weight=w_lm_head, output=argmax_in,
        grid_dim=(lm_head_grid, 1, 1), block_dim=(256, 1, 1),
    )

    # ---- Argmax ----
    mpk.argmax_partial_layer(
        input=argmax_in, output=(argmax_part_value, argmax_part_index),
        grid_dim=argmax_partial_grid, block_dim=(256, 1, 1),
    )
    mpk.argmax_reduce_layer(
        input=(argmax_part_value, argmax_part_index), output=argmax_out,
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )

    # ------------------------------------------------------------------
    # Generate task graph JSON (for inspection) and compile
    # ------------------------------------------------------------------
    results = mpk.generate_task_graph()
    import json
    tg = json.loads(results["json_file"])
    n_tasks = len(tg.get("all_tasks", []))
    n_res = len(tg.get("resident_tasks", []))
    n_data = len(tg.get("all_data", []))
    n_edges = len(tg.get("data_edges", []))
    sv = tg.get("schema_version", 1)
    n_stream = sum(1 for r in tg.get("resident_tasks", [])
                   if r.get("execution_kind", 0) == 1)
    print(f"Task graph: {n_tasks} tasks, {n_res} resident_tasks "
          f"({n_stream} streaming), {n_data} data, {n_edges} edges, "
          f"schema_v{sv}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "task_graph_rank0.json"),
                  "w") as f:
            f.write(results["json_file"])
        with open(os.path.join(args.output_dir, "test_rank0.cu"), "w") as f:
            f.write(results["cuda_code"])

    mpk.compile(output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    print(f"tokens = {tokens}")
    gen_len = step.max().item() + 1 - prompt_lengths[0].item()
    total_steps = max(1, step.max().item() + 1)
    print(f"Prompt length {prompt_lengths[0].item()}, "
          f"generate length {gen_len}, "
          f"per-token latency: {run_time / total_steps:.3f} ms")
