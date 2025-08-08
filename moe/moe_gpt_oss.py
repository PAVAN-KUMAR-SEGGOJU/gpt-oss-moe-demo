import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class MoEGPTOSS(nn.Module):
    """
    Minimal Mixture of Experts wrapper for GPT-OSS series models.
    Selects a subset of expert (feedforward) layers per token based on gating.
    """
    def __init__(self, model_name, num_experts=4, top_k=2):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_experts = num_experts
        self.top_k = top_k
        # For demo: create dummy experts as linear layers
        hidden_size = self.model.config.hidden_size
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, input_ids, attention_mask=None, output_expert_choices=False):
        # Get hidden states from the base model
        outputs = self.model.transformer(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
        gate_logits = self.gate(hidden_states)  # (batch, seq, num_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        # Select top_k experts per token
        topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        # For each token, only use top_k experts (weighted sum)
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](hidden_states))
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # (batch, seq, num_experts, hidden)
        # Weighted sum over top_k experts
        moe_output = torch.zeros_like(hidden_states)
        for k in range(self.top_k):
            idx = topk_idx[..., k].unsqueeze(-1).expand_as(hidden_states)
            weight = topk_vals[..., k].unsqueeze(-1)
            selected = torch.gather(expert_outputs, -2, idx.unsqueeze(-1)).squeeze(-2)
            moe_output += selected * weight
        if output_expert_choices:
            return moe_output, topk_idx
        return moe_output

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")
