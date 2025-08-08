import torch
from moe.moe_gpt_oss import MoEGPTOSS

def test_moe_forward():
    model = MoEGPTOSS("gpt2", num_experts=4, top_k=2)
    text = "Hello world!"
    inputs = model.tokenize(text)
    with torch.no_grad():
        output, expert_choices = model(inputs["input_ids"], output_expert_choices=True)
    assert output.shape == (1, inputs["input_ids"].shape[1], model.model.config.hidden_size)
    assert expert_choices.shape == (1, inputs["input_ids"].shape[1], model.top_k)
    print("Test passed: MoE forward output and expert selection shapes are correct.")

if __name__ == "__main__":
    test_moe_forward()
