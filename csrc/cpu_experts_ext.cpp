/*
 * cpu_experts_ext.cpp
 *
 * PyTorch C++ extension: fused SiLU-MLP forward for multiple CPU MoE experts
 * in a single parallel region.
 *
 * Key improvements over the Python sequential loop:
 *   1. All CPU experts computed concurrently — one at::parallel_for per phase,
 *      work units span ALL experts, so N threads are never idle waiting for
 *      one expert to finish before the next starts.
 *   2. gate_proj + up_proj fused into one pass over the input — reads x once
 *      per intermediate row instead of twice.
 *   3. bfloat16 weights upcast to float32 inline, no extra allocation.
 *   4. Only 2 fork-join barriers per token (Phase 1 + Phase 2) regardless of
 *      how many CPU experts there are, vs. 3 × n_experts previously.
 *
 * Formula per expert:
 *   out[t, h] = sum_i { silu(gate[i] · x[t]) * (up[i] · x[t]) * down[h, i] }
 *
 * Parallelism:
 *   Phase 1 — work units = union of all experts' I_e rows (combined).
 *   Phase 2 — work units = union of all experts' H rows (combined).
 *
 * Thread model: at::parallel_for (OpenMP or TBB, whatever PyTorch was built with).
 *
 * Build via torch.utils.cpp_extension.load() — see utils/cpu_ext.py.
 */

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// bfloat16 → float32 conversion
// bfloat16 is the upper 16 bits of IEEE-754 float32 with the lower 16 zeroed.
// ---------------------------------------------------------------------------
static inline float bf16_to_f32(uint16_t v) {
    // Union-based type pun: well-defined on all major compilers with -O3.
    union { uint32_t u; float f; } c;
    c.u = static_cast<uint32_t>(v) << 16;
    return c.f;
}

// ---------------------------------------------------------------------------
// SiLU activation: x * sigmoid(x)， deepseekmoe , xversemoe , qwenmoe all use SiLU in their experts' up_proj.
// ---------------------------------------------------------------------------
static inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// ---------------------------------------------------------------------------
// Find which expert owns global work-unit index w given prefix-sum offsets.
// Linear scan is optimal for n_exp ≤ ~10.
// ---------------------------------------------------------------------------
static inline int find_expert(const std::vector<int64_t>& offsets, int64_t w) {
    int e = 0;
    int last = static_cast<int>(offsets.size()) - 2;  // last valid expert index
    while (e < last && offsets[e + 1] <= w) ++e;
    return e;
}

// ---------------------------------------------------------------------------
// silu_mlp_batch_forward
//
// Args:
//   tokens_list  — [n_exp] tensors, each [T, H], float32 or bfloat16, CPU
//   gate_ws      — [n_exp] tensors, each [I_e, H], bfloat16, CPU
//   up_ws        — [n_exp] tensors, each [I_e, H], bfloat16, CPU
//   down_ws      — [n_exp] tensors, each [H, I_e], bfloat16, CPU
//
// Returns:
//   [n_exp] output tensors, each [T, H], bfloat16, CPU
// ---------------------------------------------------------------------------
std::vector<at::Tensor> silu_mlp_batch_forward(
    const std::vector<at::Tensor>& tokens_list,
    const std::vector<at::Tensor>& gate_ws,
    const std::vector<at::Tensor>& up_ws,
    const std::vector<at::Tensor>& down_ws)
{
    const int n_exp = static_cast<int>(tokens_list.size());
    TORCH_CHECK(n_exp > 0, "cpu_experts_ext: need at least one expert");
    TORCH_CHECK(gate_ws.size() == (size_t)n_exp, "gate_ws size mismatch");
    TORCH_CHECK(up_ws.size()   == (size_t)n_exp, "up_ws size mismatch");
    TORCH_CHECK(down_ws.size() == (size_t)n_exp, "down_ws size mismatch");

    // ── Input: ensure float32, contiguous ──────────────────────────────────
    std::vector<at::Tensor> x_f32(n_exp);
    for (int e = 0; e < n_exp; e++) {
        at::Tensor t = tokens_list[e];
        TORCH_CHECK(t.device().is_cpu(), "tokens must be on CPU");
        if (t.scalar_type() != at::kFloat)
            t = t.to(at::kFloat);
        x_f32[e] = t.contiguous();
    }

    // ── Prefix sums across experts for Phase 1 (I rows) and Phase 2 (H rows)
    std::vector<int64_t> I_off(n_exp + 1, 0);  // gate/up row offsets
    std::vector<int64_t> H_off(n_exp + 1, 0);  // down row offsets
    for (int e = 0; e < n_exp; e++) {
        TORCH_CHECK(gate_ws[e].scalar_type() == at::kBFloat16, "gate_ws must be bfloat16");
        TORCH_CHECK(up_ws[e].scalar_type()   == at::kBFloat16, "up_ws must be bfloat16");
        TORCH_CHECK(down_ws[e].scalar_type() == at::kBFloat16, "down_ws must be bfloat16");
        TORCH_CHECK(gate_ws[e].is_contiguous(), "gate_ws must be contiguous");
        TORCH_CHECK(up_ws[e].is_contiguous(),   "up_ws must be contiguous");
        TORCH_CHECK(down_ws[e].is_contiguous(), "down_ws must be contiguous");

        I_off[e + 1] = I_off[e] + gate_ws[e].size(0);   // I_e
        H_off[e + 1] = H_off[e] + down_ws[e].size(0);   // H
    }
    const int64_t total_I = I_off[n_exp];
    const int64_t total_H = H_off[n_exp];

    // ── Allocate intermediate buffers: [T, I_e] float32 per expert ─────────
    std::vector<at::Tensor> inter(n_exp);
    for (int e = 0; e < n_exp; e++) {
        int64_t T  = x_f32[e].size(0);
        int64_t I_e = gate_ws[e].size(0);
        inter[e] = at::empty({T, I_e}, at::kFloat);
    }

    // ── Allocate outputs: [T, H] float32 per expert ────────────────────────
    std::vector<at::Tensor> out_f32(n_exp);
    int64_t H_hidden = gate_ws[0].size(1);  // all experts share same hidden size
    for (int e = 0; e < n_exp; e++) {
        int64_t T = x_f32[e].size(0);
        out_f32[e] = at::zeros({T, H_hidden}, at::kFloat);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Phase 1 — fused gate+silu+up, parallelised over ALL experts' I rows
    //
    // For global work unit w:
    //   expert e  = find_expert(I_off, w)
    //   row    i  = w - I_off[e]
    //   inter[e][t, i] = silu(gate_e[i] · x_e[t]) * (up_e[i] · x_e[t])
    //
    // One read of x_e[t] per row (gate and up share the same x scan).
    // ════════════════════════════════════════════════════════════════════════
    at::parallel_for(0, total_I, 0, [&](int64_t ws, int64_t we) {
        for (int64_t w = ws; w < we; w++) {
            const int e  = find_expert(I_off, w);
            const int64_t i   = w - I_off[e];

            const int64_t T   = x_f32[e].size(0);
            const int64_t H   = x_f32[e].size(1);   // hidden dim
            const int64_t I_e = gate_ws[e].size(0);

            const float*    x_base   = x_f32[e].data_ptr<float>();
            const uint16_t* gate_row = reinterpret_cast<const uint16_t*>(
                                           gate_ws[e].data_ptr()) + i * H;
            const uint16_t* up_row   = reinterpret_cast<const uint16_t*>(
                                           up_ws[e].data_ptr())   + i * H;
            float*          inter_base = inter[e].data_ptr<float>();

            for (int64_t t = 0; t < T; t++) {
                const float* xt = x_base + t * H;
                float g = 0.0f, u = 0.0f;
                for (int64_t h = 0; h < H; h++) {
                    float gw = bf16_to_f32(gate_row[h]);
                    float uw = bf16_to_f32(up_row[h]);
                    g += gw * xt[h];
                    u += uw * xt[h];
                }
                inter_base[t * I_e + i] = silu(g) * u;
            }
        }
    });

    // ════════════════════════════════════════════════════════════════════════
    // Phase 2 — down projection, parallelised over ALL experts' H rows
    //
    // For global work unit w:
    //   expert e  = find_expert(H_off, w)
    //   row    h  = w - H_off[e]
    //   out[e][t, h] = inter[e][t, :] · down_e[h, :]
    // ════════════════════════════════════════════════════════════════════════
    at::parallel_for(0, total_H, 0, [&](int64_t ws, int64_t we) {
        for (int64_t w = ws; w < we; w++) {
            const int e  = find_expert(H_off, w);
            const int64_t h   = w - H_off[e];

            const int64_t T   = x_f32[e].size(0);
            const int64_t I_e = gate_ws[e].size(0);
            const int64_t H_e = down_ws[e].size(0);  // = H_hidden

            const float*    inter_base = inter[e].data_ptr<float>();
            const uint16_t* down_row   = reinterpret_cast<const uint16_t*>(
                                             down_ws[e].data_ptr()) + h * I_e;
            float*          out_base   = out_f32[e].data_ptr<float>();

            for (int64_t t = 0; t < T; t++) {
                const float* inter_t = inter_base + t * I_e;
                float acc = 0.0f;
                for (int64_t ii = 0; ii < I_e; ii++) {
                    acc += inter_t[ii] * bf16_to_f32(down_row[ii]);
                }
                out_base[t * H_e + h] = acc;
            }
        }
    });

    // ── Cast outputs back to bfloat16 ──────────────────────────────────────
    std::vector<at::Tensor> outputs(n_exp);
    for (int e = 0; e < n_exp; e++)
        outputs[e] = out_f32[e].to(at::kBFloat16);

    return outputs;
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_mlp_batch_forward", &silu_mlp_batch_forward,
          "Fused SiLU-MLP forward for multiple CPU experts in one parallel region.\n"
          "\n"
          "Args:\n"
          "  tokens_list: List[Tensor] — per-expert input tokens [T, H], cpu, float32/bf16\n"
          "  gate_ws:     List[Tensor] — gate_proj.weight [I, H], cpu, bfloat16\n"
          "  up_ws:       List[Tensor] — up_proj.weight   [I, H], cpu, bfloat16\n"
          "  down_ws:     List[Tensor] — down_proj.weight [H, I], cpu, bfloat16\n"
          "\n"
          "Returns:\n"
          "  List[Tensor] — output [T, H] per expert, cpu, bfloat16");
}
