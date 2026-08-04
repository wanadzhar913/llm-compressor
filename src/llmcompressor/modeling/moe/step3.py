from __future__ import annotations

import torch
import torch.nn.functional as F

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.utils.dev import skip_weights_initialize


class Step3ExpertMLP(torch.nn.Module):
    """
    Per-expert SwiGLU MLP used to replace Step 3.x packed MoELinear weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype
        )
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


class Step3LinearMoE(torch.nn.ModuleList):
    """
    Linearized replacement for Step3p5MoEMLP and Step3p7MoEMLP.

    The upstream models store routed expert projections as packed MoELinear
    modules with weight shape ``[num_experts, out_features, in_features]``. This
    replacement exposes each expert projection as an ``nn.Linear`` so quantization
    recipes with ``targets=["Linear"]`` can see the routed expert weights.
    """

    @classmethod
    @torch.no_grad()
    def from_experts_module(cls, original: torch.nn.Module, config):
        with skip_weights_initialize():
            module = cls(original, config)

        for expert_idx, expert in enumerate(module):
            expert.gate_proj.weight.copy_(original.gate_proj.weight[expert_idx])
            expert.up_proj.weight.copy_(original.up_proj.weight[expert_idx])
            expert.down_proj.weight.copy_(original.down_proj.weight[expert_idx])

        return module

    def __init__(self, original: torch.nn.Module, config):
        text_config = getattr(config, "text_config", config)
        num_experts = original.num_experts
        hidden_size = original.hidden_size
        moe_intermediate_size = original.moe_intermediate_size
        swiglu_limit = original.limit
        dtype = original.up_proj.weight.dtype
        super().__init__(
            [
                Step3ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                    swiglu_limit=swiglu_limit,
                    dtype=dtype,
                )
                for _ in range(num_experts)
            ]
        )

        self.num_experts = num_experts
        self.top_k = original.top_k
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.use_moe_router_bias = original.use_moe_router_bias
        self.need_fp32_gate = original.need_fp32_gate
        self.routed_scaling_factor = original.routed_scaling_factor
        self.gate = original.gate
        self.limit = swiglu_limit
        self.router_bias = getattr(original, "router_bias", None)
        self.moe_router_activation = getattr(text_config, "moe_router_activation", None)

    def router_bias_func(
        self, gating_output: torch.Tensor, topk: int, renormalize: bool
    ):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (
                torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20
            )
        return expert_topk_weight, indices

    def _route(self, hidden_states: torch.Tensor):
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32),
                self.gate.weight.t().to(torch.float32),
            )
        else:
            router_logits = self.gate(hidden_states)

        if self.use_moe_router_bias:
            routing_weights, selected_experts = self.router_bias_func(
                router_logits, self.top_k, renormalize=True
            )
        elif self.moe_router_activation == "sigmoid":
            routing_weights, selected_experts = _sigmoid_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )

        return routing_weights * self.routed_scaling_factor, selected_experts

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        routing_weights, selected_experts = self._route(hidden_states)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert in enumerate(self):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if get_calibrate_all_experts_flag():
                expert_out = expert(hidden_states)
                if len(top_x) == 0:
                    continue
                expert_out = expert_out[top_x]
            else:
                if len(top_x) == 0:
                    continue
                expert_out = expert(hidden_states[top_x])

            current_hidden_states = expert_out * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def _sigmoid_routing_function(
    gating_output: torch.Tensor, topk: int, renormalize: bool
):
    gating_output = gating_output.float()
    gate_prob = torch.sigmoid(gating_output)
    gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
    topk_prob, indices = torch.topk(gate_prob, k=topk, dim=1)
    expert_topk_weight = topk_prob
    if renormalize:
        expert_topk_weight = expert_topk_weight / torch.sum(
            expert_topk_weight, dim=-1, keepdim=True
        )
    return expert_topk_weight, indices
