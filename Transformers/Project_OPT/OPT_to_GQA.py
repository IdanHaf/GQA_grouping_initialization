from copy import deepcopy
from typing import TypeVar, overload, Callable

import torch
from torch import nn
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.utils import logging

logger = logging.get_logger(__name__)


class OPTGQAAttention(OPTAttention):
    def __init__(
            self,
            kv_heads: int,
            heads_grouping_arr,
            config,
            layer_idx=None,
    ):
        super().__init__(config, layer_idx)

        if self.num_heads % kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by kv_heads ({kv_heads})"
            )

        self.kv_heads = kv_heads
        self.heads_grouping_arr = heads_grouping_arr

        self.kv_inner_dim = self.kv_heads * self.head_dim

        self.k_proj = nn.Linear(self.embed_dim, self.kv_inner_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.kv_inner_dim, bias=self.enable_bias)

        self.group_size = self.num_heads // self.kv_heads

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions: bool = False,
            cache_position=None,
            **kwargs,
    ):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Changed to self.kv_heads instead of self.num_heads.
        key_states = key_states.view(bsz, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        # Current shapes: q - (b, num_heads, seq, dim), k and v - (b, kv_heads, seq, dim)

        # Duplicate each key/value head across its associated query heads (group_size).
        key_states, value_states = self.share_kv_across_queries(key_states, value_states)
        # Current shapes: q - (b, num_heads, seq, dim), k and v - (b, num_heads, seq, dim)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def share_kv_across_queries(self, k_states, v_states):
        """
        duplicate key_states and value_states according to heads_grouping_arr.

        :param k_states: key_states currently (b, kv_heads, seq, dim).
        :param v_states: value_states currently (b, kv_heads, seq, dim).
        :return: k_states, v_states of shape (b, num_heads, seq, dim) each. Where duplicates of k and v according to
        heads_grouping_arr, and thus creating the groups that share the same key and value matrix.
        """
        device = k_states.device
        batch_size, _, seq_len, dim = k_states.shape

        # k_out = torch.zeros((batch_size, self.num_heads, seq_len, dim), device=device)
        # v_out = torch.zeros((batch_size, self.num_heads, seq_len, dim), device=device)
        #
        # for batch in range(k_states.shape[0]):
        #     for idx, shared_head in enumerate(self.heads_grouping_arr):
        #         k_out[batch][idx] = k_states[batch][shared_head]
        #         v_out[batch][idx] = v_states[batch][shared_head]

        idx = torch.tensor(self.heads_grouping_arr, device=device)
        idx = idx.unsqueeze(0).expand(batch_size, -1)

        k_out = k_states.gather(1, idx.view(batch_size, self.num_heads, 1, 1)
                                .expand(batch_size, self.num_heads, seq_len, dim))
        v_out = v_states.gather(1, idx.view(batch_size, self.num_heads, 1, 1)
                                .expand(batch_size, self.num_heads, seq_len, dim))

        return k_out, v_out

    def mean_pool_weights(self, k_w, v_w):
        """
        Apply mean pooling on the heads according to self.heads_grouping_arr.

        :param k_w: Key weight matrix of shape (self.inner_dim, self.d_model).
        :param v_w: Value weight matrix of shape (self.inner_dim, self.d_model).
        :return: key_weight, value_weight matrices of shape (self.kv_inner_dim, self.d_model)
        """
        device = k_w.device

        k_w_t = k_w.transpose(0, 1)
        v_w_t = v_w.transpose(0, 1)

        k_w_t = k_w_t.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v_w_t = v_w_t.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        k_res = torch.zeros((self.kv_heads, k_w_t.shape[1], k_w_t.shape[2]), device=device)
        v_res = torch.zeros((self.kv_heads, v_w_t.shape[1], v_w_t.shape[2]), device=device)

        for idx, head in enumerate(self.heads_grouping_arr):
            k_res[head] += k_w_t[idx]
            v_res[head] += v_w_t[idx]

        k_res /= self.group_size
        v_res /= self.group_size

        k_res = k_res.transpose(1, 2).reshape(self.kv_inner_dim, self.embed_dim)
        v_res = v_res.transpose(1, 2).reshape(self.kv_inner_dim, self.embed_dim)

        return k_res, v_res

    @classmethod
    def from_opt_attention(cls, opt: OPTAttention, kv_heads: int, heads_grouping_arr: list[int]):
        if len(heads_grouping_arr) != opt.num_heads:
            raise ValueError("Length of heads_grouping_arr must match num_heads")

        if any(h >= kv_heads for h in heads_grouping_arr):
            raise ValueError(
                f"group in list can't have higher or equal number then {kv_heads} (first group starts at 0)"
            )

        config = opt.config

        opt_gqa_attention = cls(
            kv_heads=kv_heads,
            heads_grouping_arr=heads_grouping_arr,
            config=config,
            layer_idx=opt.layer_idx
        )

        opt_gqa_attention.q_proj.weight.data = opt.q_proj.weight.data
        # t5_gqa_attention expects k as d_model -> kv_heads*d_kv. currently k shape is d_model -> num_heads*d_kv.
        k_w, v_w = opt_gqa_attention.mean_pool_weights(opt.k_proj.weight.data, opt.v_proj.weight.data)
        opt_gqa_attention.k_proj.weight.data = k_w
        opt_gqa_attention.v_proj.weight.data = v_w

        opt_gqa_attention.out_proj.weight.data = opt.out_proj.weight.data

        return opt_gqa_attention


ModuleType = TypeVar("ModuleType", bound=nn.Module)


@overload
def convert_opt_to_gqa(
        module: ModuleType, kv_heads: int, heads_grouping_arr: list[list[int]], inplace: bool = False,
        _counter: list[int] = None
) -> ModuleType:
    ...


@overload
def convert_opt_to_gqa(
        module: OPTAttention, kv_heads: int, heads_grouping_arr: list[list[int]], inplace: bool = False,
        _counter: list[int] = None
) -> OPTGQAAttention:
    ...


def convert_opt_to_gqa(
        module, kv_heads: int, heads_grouping_arr: list[list[int]], inplace: bool = False, _counter: list[int] = None
):
    if _counter is None:
        _counter = [0]

    if isinstance(module, OPTAttention):
        idx = _counter[0]
        _counter[0] += 1
        return OPTGQAAttention.from_opt_attention(
            module,
            kv_heads=kv_heads,
            heads_grouping_arr=heads_grouping_arr[idx],
        )

    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        out._modules[name] = convert_opt_to_gqa(child,
                                                kv_heads=kv_heads,
                                                heads_grouping_arr=heads_grouping_arr,
                                                inplace=True,
                                                _counter=_counter,
                                                )
    return out

