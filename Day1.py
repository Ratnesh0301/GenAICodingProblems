"""
Coding problem: Token cost estimator using tiktoken
easy
Build a utility that counts tokens for a prompt+completion pair and estimates cost for GPT-4o. Critical skill for production cost management."""

import tiktoken

def estimate_count(prompt:str, completion:str, model:str="gpt-4o") -> dict:
    """Estimate token count and cost for a prompt+completion pair."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(completion))
    total_tokens = prompt_tokens + completion_tokens

    # GPT-4o pricing (as of 2026-04-14)
    # Input: $0.0025 / 1M tokens
    # Output: $0.012 / 1M tokens
    pricing = {
        "gpt-4o": {
            "input_per_1m": 2.5,
            "output_per_1m": 12.0
        }
    }

    if model not in pricing:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "error": "Model pricing not found. Please update the pricing dictionary."
        }

    model_pricing = pricing[model]
    input_cost = (prompt_tokens / 1_000_000) * model_pricing["input_per_1m"]
    output_cost = (completion_tokens / 1_000_000) * model_pricing["output_per_1m"]
    total_cost = input_cost + output_cost

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# Example usage
prompt = "What is the capital of France?"
completion = "The capital of France is Paris."

result = estimate_count(prompt, completion)
print(f"Prompt: {prompt}")
print(f"Completion: {completion}")
print(f"Prompt tokens: {result['prompt_tokens']}")
print(f"Completion tokens: {result['completion_tokens']}")
print(f"Total tokens: {result['total_tokens']}")
print(f"Input cost: ${result['input_cost']:.6f}")
print(f"Output cost: ${result['output_cost']:.6f}")
print(f"Total cost: ${result['total_cost']:.6f}")
    