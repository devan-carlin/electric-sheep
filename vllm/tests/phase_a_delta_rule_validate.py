"""Validate my pure-Python gated-delta-rule recurrent loop against vLLM's
CPU reference `chunk_gated_delta_rule_cpu`.

If they match -> my reference is correct -> the XPU kernel's core output is
the bug. If they differ -> my reference is wrong (fix it, re-test kernel).

Standalone, no model load. CPU only.
"""

import torch
import torch.nn.functional as F
import vllm._custom_ops as ops

T = 8
H = 48      # v-heads
HK = 128    # k dim
HV = 128    # v dim
DEV = "cpu"


def my_recurrent(q, k, v, g, beta):
    """q,k,v: [T,H,dim] (l2-normalized). g: [T,H] (log-decay, negative).
    beta: [T,H]. Returns [T,H,HV] raw recurrent output (before z-gate).
    State: [H, k_dim, v_dim].
    """
    T = q.size(0)
    state = torch.zeros(H, HK, HV, device=DEV, dtype=torch.float32)
    outs = []
    for t in range(T):
        qt = q[t].float()                       # [H, HK]
        kt = k[t].float()                       # [H, HK]
        vt = v[t].float()                       # [H, HV]
        gt = g[t].float()                       # [H]
        bt = beta[t].float().unsqueeze(-1)      # [H, 1]
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)  # [H,1,1]
        pred = torch.bmm(state, vt.unsqueeze(-1)).squeeze(-1)  # [H,HV]
        delta = bt * (vt - pred)                # [H,HV]
        state = state * decay + torch.bmm(
            kt.unsqueeze(-1), delta.unsqueeze(1))  # [H,HK,HV]
        out_t = torch.bmm(state, qt.unsqueeze(-1)).squeeze(-1)  # [H,HV]
        outs.append(out_t)
    return torch.stack(outs, dim=0)             # [T,H,HV]


def main():
    torch.manual_seed(0)
    # l2-normalized q, k
    q = F.normalize(torch.randn(T, H, HK, device=DEV), dim=-1)
    k = F.normalize(torch.randn(T, H, HK, device=DEV), dim=-1)
    v = torch.randn(T, H, HV, device=DEV)
    # g: log-decay, negative (like -exp(A_log)*softplus(a+dt_bias))
    g = -torch.rand(T, H, device=DEV) * 2.0 - 0.1
    beta = torch.rand(T, H, device=DEV)  # in (0,1)

    # My reference
    mine = my_recurrent(q, k, v, g, beta)

    # vLLM CPU reference
    initial_state = torch.zeros(1, H, HK, HV, device=DEV, dtype=torch.float32)
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    idx = torch.tensor([0], dtype=torch.int32, device=DEV)
    ref, _ = ops.chunk_gated_delta_rule_cpu(
        query=q, key=k, value=v, g=g, beta=beta,
        initial_state=initial_state, output_final_state=True,
        cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True,
        initial_state_indices=idx,
    )
    ref = ref.squeeze(0)  # [T,H,HV]

    def cos(a, b):
        a = a.float().flatten(); b = b.float().flatten()
        n = min(a.numel(), b.numel())
        return F.cosine_similarity(a[:n], b[:n], dim=0).item()

    print(f"cos(mine, cpu_ref) = {cos(mine, ref):+.6f}")
    print(f"mine[:6]  = {mine.flatten()[:6].tolist()}")
    print(f"ref[:6]   = {ref.flatten()[:6].tolist()}")
    print(f"mine norm = {mine.norm().item():.4f}   ref norm = {ref.norm().item():.4f}")
    if cos(mine, ref) > 0.99:
        print("PASS: my pure-Python reference matches vLLM CPU reference.")
        print("-> reference is correct; XPU kernel core output is the bug.")
    else:
        print("MISMATCH: my reference differs from vLLM CPU reference.")
        print("-> fix the reference before trusting the kernel verdict.")


if __name__ == "__main__":
    main()