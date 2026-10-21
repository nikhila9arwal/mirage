#pragma once
#include <algorithm>
#include <climits>
#include <initializer_list>

// Host validation for the deliberately small prototype. The native graph keeps
// truthful per-slot tensor views; only these validated views create the physical
// batch template. This does not relax ordinary ownership or dependency checks.
static void *experimental_moe_host_maps[EXPERIMENTAL_BATCH_OPERATORS] = {};
static int32_t *experimental_moe_host_data_to_op = nullptr;

static unsigned long long experimental_moe_unsigned_env(
    char const *name, unsigned long long fallback) {
  char const *text = std::getenv(name);
  if (text == nullptr) return fallback;
  if (*text == '\0') throw std::runtime_error(std::string("Empty ") + name);
  unsigned long long value = 0;
  for (char const *c = text; *c; c++) {
    if (*c < '0' || *c > '9' || value > (ULLONG_MAX - (*c - '0')) / 10ull)
      throw std::runtime_error(std::string("Invalid unsigned value for ") + name);
    value = value * 10ull + (*c - '0');
  }
  return value;
}

static void init_experimental_moe(
    ResidentRuntimeConfig &config,
    std::vector<TaskDesc> const &tasks,
    std::vector<FullDataDesc> const &data,
    std::vector<uint32_t> const &slots,
    std::vector<TaskId> const &data_to_task) {
  bool found = false, marked_dense = false;
  for (auto const &task : tasks) {
    int kind = experimental_moe_kind(task.task_type);
    found = found || kind >= 0;
    marked_dense = marked_dense || kind == EXPERIMENTAL_BATCH_QKV ||
                   kind == EXPERIMENTAL_BATCH_WO;
  }
  if (!found) return;
#ifdef NAS_MOE_OCCUPANCY
  // This optional recorder still describes W13/W2 only. Reject marked graphs
  // even with dense batching off, keeping the four comparison modes matched.
  if (marked_dense)
    throw std::runtime_error("NAS_MOE_OCCUPANCY does not support marked dense projections; use task/batch tracing instead");
#endif
  if (config.num_request_slots != EXPERIMENTAL_MOE_STATE_SLOTS ||
      config.num_contexts < 0 || !config.context_input_is_int64 ||
      config.num_gpus != 1 || config.num_local_schedulers != 1 ||
      config.num_workers < 1 || config.num_workers > 16 ||
      config.num_remote_schedulers != 0 || !config.split_worker_scheduler ||
      get_resident_execution_mode(config) != RESIDENT_EXECUTION_SCHEDULER_DISPATCH ||
      tasks.size() >= EXPERIMENTAL_MOE_ITEM_TAG ||
      data.size() >= EXPERIMENTAL_MOE_ITEM_TAG ||
      slots.size() != data.size() || data_to_task.size() != data.size()) {
    throw std::runtime_error("Experimental batching requires the compiled state-slot count, 1..16 workers, finite scalar ContextIO, one split scheduler, single GPU resident scheduler_dispatch");
  }
  ExperimentalMoeState host = {};
  auto early = experimental_moe_unsigned_env("MIRAGE_MOE_EARLY_COALESCE", 0);
  if (early > 1u) throw std::runtime_error("MIRAGE_MOE_EARLY_COALESCE must be 0 or 1");
  host.early_coalesce = static_cast<uint32_t>(early);
  auto deadline = experimental_moe_unsigned_env("MIRAGE_MOE_DEADLINE_COALESCE", 0);
  if (deadline > 1u) throw std::runtime_error("MIRAGE_MOE_DEADLINE_COALESCE must be 0 or 1");
  host.deadline_coalesce = static_cast<uint32_t>(deadline);
#ifdef NAS_MOE_OCCUPANCY
  // The archived recorder assumes one expert per physical batch.
  if (early || deadline) throw std::runtime_error("Coalescing requires batch tracing; the old occupancy recorder is unsupported");
#endif
#ifdef NAS_INVESTIGATE_ALL_ROWS
  if (early || deadline) throw std::runtime_error("Coalescing is only supported with independent progress");
#endif
  char const *setting = std::getenv("MIRAGE_DENSE_BATCHING");
  std::string dense_mode = setting == nullptr ? "none" : setting;
  if (dense_mode != "none" && dense_mode != "qkv" &&
      dense_mode != "wo" && dense_mode != "both")
    throw std::runtime_error("MIRAGE_DENSE_BATCHING must be none, qkv, wo, or both");
  auto moe_rows = experimental_moe_unsigned_env("MIRAGE_MOE_BATCH_ROWS", 4);
  auto moe_us = experimental_moe_unsigned_env("MIRAGE_MOE_TIMEOUT_US", 100);
  auto dense_rows = experimental_moe_unsigned_env("MIRAGE_DENSE_BATCH_ROWS", moe_rows);
  auto dense_us = experimental_moe_unsigned_env("MIRAGE_DENSE_TIMEOUT_US", moe_us);
  if (moe_rows == 0 || moe_rows > EXPERIMENTAL_MOE_BATCH_CAPACITY ||
      dense_rows == 0 || dense_rows > EXPERIMENTAL_MOE_BATCH_CAPACITY ||
      moe_us > 1000000ull || dense_us > 1000000ull)
    throw std::runtime_error("Experimental batching expects batch rows 1..16 and timeout 0..1000000 us");
  uint32_t count[EXPERIMENTAL_BATCH_OPERATORS][EXPERIMENTAL_MOE_STATE_SLOTS] = {};
  void *weights[EXPERIMENTAL_BATCH_OPERATORS] = {};
  void *routing[EXPERIMENTAL_BATCH_OPERATORS] = {};
  std::vector<int32_t> data_to_op(data.size(), -1);
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; op_id++) {
    uint32_t kind = op_id % EXPERIMENTAL_BATCH_KINDS;
    auto &op = host.operators[op_id];
    op.kind = kind;
    op.layer_id = op_id / EXPERIMENTAL_BATCH_KINDS;
    op.resident_task_id = RESIDENT_TASK_INVALID_ID;
    bool dense = kind == EXPERIMENTAL_BATCH_QKV || kind == EXPERIMENTAL_BATCH_WO;
    op.enabled = !dense || dense_mode == "both" ||
        (kind == EXPERIMENTAL_BATCH_QKV && dense_mode == "qkv") ||
        (kind == EXPERIMENTAL_BATCH_WO && dense_mode == "wo");
    op.input_width = kind == EXPERIMENTAL_BATCH_W2 ? EXPERIMENTAL_MOE_I : EXPERIMENTAL_MOE_H;
    op.output_width = kind == EXPERIMENTAL_BATCH_W13 ? 2 * EXPERIMENTAL_MOE_I :
        (kind == EXPERIMENTAL_BATCH_QKV ? 3 * EXPERIMENTAL_MOE_H / 2 : EXPERIMENTAL_MOE_H);
    op.num_experts = dense ? 1 : EXPERIMENTAL_MOE_EXPERTS;
    op.threshold = dense ? dense_rows : moe_rows;
    op.timeout_ns = 1000ull * (dense ? dense_us : moe_us);
  }

  // Native resident_data assigns a different family to each registered graph
  // operator. Sort those identities, then number each kind's families in order.
  // The demo registers complete layers in order and checks the resulting labels.
  // Grouping uses the family binding, never a shared TaskType or weight pointer.
  struct Family { uint32_t resident_task_id, kind; };
  std::vector<Family> families;
  for (size_t d = 0; d < data.size(); d++) {
    if (data_to_task[d] >= tasks.size() || slots[d] >= EXPERIMENTAL_MOE_STATE_SLOTS)
      throw std::runtime_error("Experimental batching has an invalid task or state-slot mapping");
    int kind = experimental_moe_kind(tasks[data_to_task[d]].task_type);
    if (kind < 0) continue;
    uint32_t family_id = data[d].resident_task_id;
    if (family_id >= static_cast<uint32_t>(config.num_resident_tasks))
      throw std::runtime_error("Experimental batching requires a native resident family ID");
    auto family = std::find_if(families.begin(), families.end(), [&](Family const &f) {
      return f.resident_task_id == family_id;
    });
    if (family == families.end()) {
      families.push_back({family_id, static_cast<uint32_t>(kind)});
    } else if (family->kind != static_cast<uint32_t>(kind)) {
      throw std::runtime_error("One native family contains different batch operation kinds");
    }
  }
  std::sort(families.begin(), families.end(), [](Family const &a, Family const &b) {
    return a.resident_task_id < b.resident_task_id;
  });
  uint32_t families_per_kind[EXPERIMENTAL_BATCH_KINDS] = {};
  for (auto const &family : families) {
    uint32_t layer = families_per_kind[family.kind]++;
    if (layer >= EXPERIMENTAL_MOE_LAYERS)
      throw std::runtime_error("Native graph has more operations per kind than NAS_MOE_LAYERS");
    auto &op = host.operators[layer * EXPERIMENTAL_BATCH_KINDS + family.kind];
    op.resident_task_id = family.resident_task_id;
    op.present = 1;
  }
  auto tensor = [](TensorDesc const &t, int dtype, std::initializer_list<int> dims) {
    if (t.data_type != dtype || t.num_dims != static_cast<int>(dims.size()) || t.base_ptr == nullptr)
      throw std::runtime_error("Experimental batch tensor dtype/rank mismatch");
    std::vector<int> shape(dims);
    size_t stride = 1;
    for (int axis = shape.size() - 1; axis >= 0; axis--) {
      if (t.dim[axis] != shape[axis] || t.stride[axis] != stride)
        throw std::runtime_error("Experimental batching requires exact contiguous compiled-shape views");
      stride *= shape[axis];
    }
  };
  auto same_pointer = [](void *&base, void *value) {
    if (base != nullptr && base != value)
      throw std::runtime_error("Rows of one native family must share input, output, routing, and weight allocations");
    base = value;
  };
  for (size_t d = 0; d < data.size(); d++) {
    int kind = experimental_moe_kind(tasks[data_to_task[d]].task_type);
    if (kind < 0) continue;
    int32_t op_id = -1;
    for (uint32_t candidate = 0; candidate < EXPERIMENTAL_BATCH_OPERATORS; candidate++) {
      auto const &op = host.operators[candidate];
      if (op.present && op.resident_task_id == data[d].resident_task_id) {
        op_id = candidate;
        break;
      }
    }
    if (op_id < 0)
      throw std::runtime_error("Batch row has no concrete native family binding");
    auto &op = host.operators[op_id];
    uint32_t slot = slots[d];
    if (++count[op_id][slot] != 1 || data[d].initial_predecessor_count <= 0)
      throw std::runtime_error("Experimental batching requires one non-root logical node per present operation and slot");
    int K = op.input_width, N = op.output_width;
    auto const &x = data[d].inputs[0];
    auto const &w = data[d].inputs[1];
    auto const &y = data[d].outputs[0];
    tensor(x, mirage::type::DT_BFLOAT16, {1, K});
    tensor(y, mirage::type::DT_BFLOAT16, {1, N});
    if (op.num_experts == 1) {
      tensor(w, mirage::type::DT_BFLOAT16, {N, K});
    } else {
      tensor(w, mirage::type::DT_BFLOAT16, {int(EXPERIMENTAL_MOE_EXPERTS), N, K});
      auto const &r = data[d].inputs[2];
      tensor(r, mirage::type::DT_INT32, {1, int(EXPERIMENTAL_MOE_EXPERTS)});
      same_pointer(routing[op_id], static_cast<char *>(r.base_ptr) - slot * EXPERIMENTAL_MOE_EXPERTS * 4);
      op.routing = static_cast<int32_t const *>(routing[op_id]);
    }
    same_pointer(op.input, static_cast<char *>(x.base_ptr) - slot * K * 2);
    same_pointer(op.output, static_cast<char *>(y.base_ptr) - slot * N * 2);
    same_pointer(weights[op_id], w.base_ptr);
    data_to_op[d] = op_id;
  }
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; op_id++) {
    auto const &op = host.operators[op_id];
    if (!op.present) continue;
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; slot++)
      if (count[op_id][slot] != 1)
        throw std::runtime_error("Experimental batching is missing a logical operation/slot row");
  }
  if ((dense_mode == "qkv" || dense_mode == "both") &&
      !host.operators[EXPERIMENTAL_BATCH_QKV].present)
    throw std::runtime_error("QKV batching requested but the graph has no marked QKV operation");
  if ((dense_mode == "wo" || dense_mode == "both") &&
      !host.operators[EXPERIMENTAL_BATCH_WO].present)
    throw std::runtime_error("W_O batching requested but the graph has no marked W_O operation");

  // All bindings are validated before allocation. Disabled dense operations
  // still activate this controller and retain their ordinary native task body.
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; op_id++) {
    auto &op = host.operators[op_id];
    if (!op.present) continue;
    CUtensorMap map;
    uint64_t shape[2] = {uint64_t(op.num_experts) * op.output_width, op.input_width};
    uint64_t stride[2] = {1, op.input_width};
    uint32_t tile[2] = {64, 64};
    // Dense E=1 describes exactly [N,K]. Both paths reference the original
    // shared weights; batching copies neither weights nor activation vectors.
    mirage::runtime::fill_tma_desc<mirage::runtime::bfloat16, 3, 3, 3, 2>(
        &map, weights[op_id], shape, stride, tile, 1, 1);
    CUDA_CHECK(cudaMalloc(&experimental_moe_host_maps[op_id], sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(experimental_moe_host_maps[op_id], &map, sizeof(map), cudaMemcpyHostToDevice));
    op.weight_map = experimental_moe_host_maps[op_id];
    fprintf(stderr, "BATCH_OPERATOR op_id=%u kind=%u layer=%u resident_task_id=%u enabled=%u K=%u N=%u E=%u threshold=%u timeout_ns=%llu\n",
            op_id, op.kind, op.layer_id, op.resident_task_id, op.enabled, op.input_width,
            op.output_width, op.num_experts, op.threshold, op.timeout_ns);
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&experimental_moe_host_data_to_op),
                        data_to_op.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(experimental_moe_host_data_to_op, data_to_op.data(),
                        data_to_op.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  host.data_to_op_id = experimental_moe_host_data_to_op;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&config.experimental_moe), sizeof(host)));
  CUDA_CHECK(cudaMemcpy(config.experimental_moe, &host, sizeof(host), cudaMemcpyHostToDevice));
  fprintf(stderr, "EXPERIMENTAL_MOE enabled: state_slots=%u batch_capacity=%u workers=%d descriptors=%u dense=%s H=%u I=%u queues(worker,completion,control)=(%llu,%llu,%llu)\n",
          EXPERIMENTAL_MOE_STATE_SLOTS, EXPERIMENTAL_MOE_BATCH_CAPACITY,
          config.num_workers, EXPERIMENTAL_MOE_POOL, dense_mode.c_str(),
          EXPERIMENTAL_MOE_H, EXPERIMENTAL_MOE_I,
          static_cast<unsigned long long>(config.per_worker_queue_len),
          static_cast<unsigned long long>(config.per_completion_queue_len),
          static_cast<unsigned long long>(config.per_sched_queue_len));
}

static void finalize_experimental_moe(ResidentRuntimeConfig &config) {
  if (config.experimental_moe == nullptr) return;
  CUDA_CHECK(cudaFree(config.experimental_moe));
  config.experimental_moe = nullptr;
  CUDA_CHECK(cudaFree(experimental_moe_host_data_to_op));
  experimental_moe_host_data_to_op = nullptr;
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; op_id++) {
    CUDA_CHECK(cudaFree(experimental_moe_host_maps[op_id]));
    experimental_moe_host_maps[op_id] = nullptr;
  }
}
