import torch
import matplotlib.pyplot as plt
from moe.moe_gpt_oss import MoEGPTOSS

# Change to a real GPT-OSS model name if available
MODEL_NAME = "gpt2"  # Placeholder for GPT-OSS series

def visualize_expert_selection(token_ids, expert_choices, tokenizer):
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    plt.figure(figsize=(10, 3))
    for i, experts in enumerate(expert_choices[0]):
        plt.scatter([i]*len(experts), experts, label=f"Token: {tokens[i]}")
    plt.xlabel("Token Position")
    plt.ylabel("Expert Index")
    plt.title("Top-k Expert Selection per Token")
    plt.show()

def main():
    moe_model = MoEGPTOSS(MODEL_NAME, num_experts=4, top_k=2)
    text = "The quick brown fox jumps over the lazy dog."
    inputs = moe_model.tokenize(text)
    with torch.no_grad():
        moe_output, expert_choices = moe_model(inputs["input_ids"], output_expert_choices=True)
    print("Token IDs:", inputs["input_ids"][0].tolist())
    print("Expert choices per token:", expert_choices[0].tolist())
    visualize_expert_selection(inputs["input_ids"][0], expert_choices, moe_model.tokenizer)

if __name__ == "__main__":
    main()
