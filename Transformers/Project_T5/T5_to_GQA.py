from copy import deepcopy
from typing import TypeVar, overload

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Attention
from transformers.models.t5.configuration_t5 import T5Config


class T5GQAAttention(T5Attention):
    def __init__(
            self,
            kv_heads: int,
            config,
            has_relative_attention_bias=False,
            layer_idx=None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)

        if self.n_heads % kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by kv_heads ({kv_heads})"
            )

        self.kv_heads = kv_heads
        self.kv_inner_dim = self.kv_heads * self.key_value_proj_dim

        self.k = nn.Linear(self.d_model, self.kv_inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.kv_inner_dim, bias=False)

        self.group_size = self.n_heads // self.kv_heads

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            # Changed to self.kv_heads instead of self.n_heads.
            key_states = key_states.view(batch_size, -1, self.kv_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.kv_heads, self.key_value_proj_dim).transpose(1, 2)

            # Current shapes: q - (b, n_heads, seq, dim), k and v - (b, kv_heads, seq, dim)
            # Duplicate each key/value head across its associated query heads (group_size).
            # This is required for GQA: same k/v across grouped query heads.
            key_states = key_states.repeat(1, 1, self.group_size, 1).reshape(batch_size, self.n_heads, -1,
                                                                             self.key_value_proj_dim)
            value_states = value_states.repeat(1, 1, self.group_size, 1).reshape(batch_size, self.n_heads, -1,
                                                                                 self.key_value_proj_dim)
            # Current shapes: q - (b, n_heads, seq, dim), k and v - (b, n_heads, seq, dim)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states),
        # compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    def aggregate_weights(self, k_w, v_w):
        """
        Apply mean pooling on the heads.

        :param k_w: Key weight matrix of shape (self.inner_dim, self.d_model).
        :param v_w: Value weight matrix of shape (self.inner_dim, self.d_model).
        :return: key_weight, value_weight matrices of shape (self.kv_inner_dim, self.d_model)
        """
        device = k_w.device
        k_w_agg = torch.zeros(self.kv_inner_dim, self.d_model, device=device)
        v_w_agg = torch.zeros(self.kv_inner_dim, self.d_model, device=device)

        for kv_head in range(self.kv_heads):
            for col in range(self.key_value_proj_dim):
                for head in range(self.group_size):
                    k_w_agg[kv_head * self.key_value_proj_dim + col] += k_w[
                        kv_head * self.key_value_proj_dim * self.group_size + head * self.key_value_proj_dim + col
                        ]
                    v_w_agg[kv_head * self.key_value_proj_dim + col] += v_w[
                        kv_head * self.key_value_proj_dim * self.group_size + head * self.key_value_proj_dim + col
                        ]
                k_w_agg[kv_head * self.key_value_proj_dim + col] /= self.group_size
                v_w_agg[kv_head * self.key_value_proj_dim + col] /= self.group_size

        return k_w_agg, v_w_agg

    @classmethod
    def from_t5_attention(cls, t5: T5Attention, kv_heads: int):
        config = T5Config(d_model=t5.d_model, d_kv=t5.key_value_proj_dim, num_heads=t5.n_heads,
                          relative_attention_num_buckets=t5.relative_attention_num_buckets,
                          relative_attention_max_distance=t5.relative_attention_max_distance,
                          dropout_rate=t5.dropout, is_decoder=t5.is_decoder)
        t5_gqa_attention = cls(
            kv_heads=kv_heads,
            config=config,
            has_relative_attention_bias=t5.has_relative_attention_bias,
            layer_idx=t5.layer_idx
        )

        # Copy all of the weights verbatim from the original T5Attention module.
        # NOTE: In the T5 GQA implementation, all of the attention head aggregations
        # happen in the 'forward' method.  The weights themselves are not modified.
        t5_gqa_attention.q.weight.data = t5.q.weight.data
        # t5_gqa_attention expects k as d_model -> kv_heads*d_kv. currently k shape is d_model -> n_heads*d_kv.
        k_w, v_w = t5_gqa_attention.aggregate_weights(t5.k.weight.data, t5.v.weight.data)
        t5_gqa_attention.k.weight.data = k_w
        t5_gqa_attention.v.weight.data = v_w

        t5_gqa_attention.o.weight.data = t5.o.weight.data
        if t5.has_relative_attention_bias:
            t5_gqa_attention.relative_attention_bias.weight.data = (
                t5.relative_attention_bias.weight.data
            )

        return t5_gqa_attention


ModuleType = TypeVar("ModuleType", bound=nn.Module)


@overload
def convert_t5_to_gqa(
        module: ModuleType, kv_heads: int, inplace: bool = False
) -> ModuleType:
    ...


@overload
def convert_t5_to_gqa(
        module: T5Attention, kv_heads: int, inplace: bool = False
) -> T5GQAAttention:
    ...


def convert_t5_to_gqa(module, kv_heads: int, inplace: bool = False):
    if isinstance(module, T5Attention):
        return T5GQAAttention.from_t5_attention(module, kv_heads=kv_heads)

    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        out._modules[name] = convert_t5_to_gqa(child, kv_heads=kv_heads, inplace=True)
    return out
