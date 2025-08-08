# GPT-OSS Mixture of Experts (MoE) Demo

This project demonstrates a Mixture of Experts (MoE) approach using GPT-OSS series models. It includes a minimal MoE wrapper, example usage, and visualizations of expert selection per token.

## Features
- Minimal MoE implementation for GPT-OSS models
- Example scripts showing expert/layer selection per token
- Ready-to-run and GitHub-deployable

## Getting Started

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the example: `python examples/moe_example.py`

## Project Structure
- `moe/` - MoE wrapper and core logic
- `examples/` - Example scripts
- `tests/` - Unit tests

## Requirements
- Python 3.8+
- torch
- transformers (for GPT-OSS or similar models)

## License
MIT
