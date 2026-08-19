import math

import torch
import triton
import triton.language as tl

class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            is_causal: bool = False,
    ) -> torch.Tensor:

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        scale = 1 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty(
            (batch_size, n_queries),
            device=Q.device,
            dtype=Q.dtype,
        )

        for batch_idx in range(batch_size):
            for q_start in range(0, n_queries, Q_TILE_SIZE):
                q_end = min(q_start + Q_TILE_SIZE, n_queries)

                Q_i = Q[batch_idx, q_start:q_end, :]
                current_q_size = Q_i.shape(0)

                O_i = torch.zeros((current_q_size, d), device=Q.device, dtype=Q.dtype)
                l_i = torch.zeros(current_q_size, device=Q.device, dtype=Q.dtype)
                m_i = torch.full((current_q_size,),float('-inf'),device=Q.device, dtype=Q.dtype)

                for k_start in range(0, n_keys, K_TILE_SIZE):
                    k_end = min(k_start + K_TILE_SIZE, n_keys)

                    K_j = K[batch_idx, k_start:k_end, :]
                    V_j = V[batch_idx, k_start:k_end, :]

                    S_ij = Q_i @ K_j.transpose(-2, -1)
                    S_ij = S_ij * scale

                    if is_causal:
                        q_indices = torch.arange(q_start, q_end, device = Q.device)
                        k_indices = torch.arange(k_start, k_end, device = Q.device)

                        causal_mask = k_indices[None, :] > q_indices[:, None]

                        S_ij = S_ij + causal_mask *(-1e6)

                    m_new = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)

                    P_tiled = torch.exp(S_ij - m_new[:, None])

                    correction = torch.exp(m_i - m_new)

                    l_i = (correction * l_i + torch.sum(P_tiled, dim=-1))

                    O_i = (correction[:, None] * O_i + P_tiled @ V_j)

                    m_i = m_new

                O_i = O_i / l_i[:, None]
                L_i = m_i + torch.log(l_i)
                O[batch_idx, q_start:q_end, :] = O_i
                L[batch_idx, q_start:q_end] = L_i
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_pytorch_compiled(Q, K, V, O, grad_output, L, ctx.is_causal)

        return dQ, dK, dV, None

def flash_backward_pytorch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        O: torch.Tensor,
        dO: torch.Tensor,
        L: torch.Tensor,
        is_causal: bool = False,
):

    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)

    D = torch.sum(O * dO, dim=-1)
    S = Q @ K.transpose(-2, -1)
    S = S * scale
    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]

        q_indices = torch.arange(
            n_queries,
            device=Q.device,
        )
        k_indices = torch.arange(
            n_keys,
            device=Q.device,
        )

        causal_mask = (
                k_indices[None, :]
                > q_indices[:, None]
        )

        S = S + causal_mask * (-1e6)

    P = torch.exp(S - L[..., None])

    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)

    dS = P * (dP - D[..., None])
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale

    return dQ, dK, dV

flash_backward_pytorch_compiled = torch.compile(flash_backward_pytorch)


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_indices = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))

    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    for _ in range(0, N_KEYS, K_TILE_SIZE):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            k_indices = (_ + tl.arange(0, K_TILE_SIZE))
            causal_mask = (k_indices[None, :] > q_indices[:, None])
            S_ij = tl.where(causal_mask, -1e6, 0.0)

        current_m = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, current_m)

        correction = tl.exp(m_i - m_new)

        P_tiled = tl.exp(S_ij - m_new[:, None])
        l_i = (correction * l_i + tl.sum(P_tiled, axis=1))

        O_i = correction[:, None] * O_i
        P_tiled = P_tiled.to(V_j.dtype)
        O_i = tl.dot(P_tiled, V_j, acc=O_i)

        m_i = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    O_i = O_i.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            is_causal: bool=False,
    ) -> torch.Tensor:

        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)
        L = torch.empty(
            (batch_size, n_queries),
            dtype=Q.dtype,
            device=Q.device,
        )

        scale = 1 / math.sqrt(d)

        grid = (
            triton.cdiv(n_queries, Q_TILE_SIZE),
            batch_size
        )

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,

            Q.stride(0),
            Q.stride(1),
            Q.stride(2),

            K.stride(0),
            K.stride(1),
            K.stride(2),

            V.stride(0),
            V.stride(1),
            V.stride(2),

            O.stride(0),
            O.stride(1),
            O.stride(2),

            L.stride(0),
            L.stride(1),

            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,

            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_pytorch_compiled(Q, K, V, O, grad_output, L, ctx.is_causal)

        return dQ, dK, dV, None

flash_attention_triton = FlashAttentionTriton.apply

    

    