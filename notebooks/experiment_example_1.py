import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def compute_step_metrics(logits: torch.Tensor):
    """
    Given logits for a single generation step (shape [vocab_size]), compute:
    - entropy
    - top-1 probability
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
    top1_prob = torch.max(probs).item()
    return {"entropy": entropy, "top1_prob": top1_prob}

def generate_and_analyze(model, tokenizer, prompt: str, max_new_tokens: int = 20):
    """
    Generate from `prompt` and record per-step metrics.
    Returns the generated text and a list of dicts (one per new token).
    """
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate with scores
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    
    # `scores` is a tuple of logits for each step
    scores = outputs.scores  # Tuple[Tensor], each shape [batch_size, vocab_size]
    
    token_ids = outputs.sequences[0, input_ids.size(-1):].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    metrics_per_step = []
    for step_logits, token in zip(scores, tokens):
        # step_logits: [1, vocab_size] → squeeze to [vocab_size]
        m = compute_step_metrics(step_logits.squeeze(0))
        m["token"] = token
        metrics_per_step.append(m)
    
    gen_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return gen_text, metrics_per_step

def summarize_metrics(metrics_list):
    """
    Given a list of step‐metrics dicts, compute mean and variance for each metric.
    """
    import statistics
    entropies = [m["entropy"] for m in metrics_list]
    top1s     = [m["top1_prob"] for m in metrics_list]
    
    summary = {
        "steps": len(metrics_list),
        "mean_entropy": statistics.mean(entropies),
        "stdev_entropy": statistics.stdev(entropies) if len(entropies)>1 else 0.0,
        "mean_top1_prob": statistics.mean(top1s),
        "stdev_top1_prob": statistics.stdev(top1s) if len(top1s)>1 else 0.0,
    }
    return summary

if __name__ == "__main__":
    # Example with GPT-2
    for model_name in ["gpt2", "huggyllama/llama-7b"]:
        print(f"\n=== Analyzing {model_name} ===")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        prompt = "In a distant future, machines and humans"
        gen_text, metrics = generate_and_analyze(model, tokenizer, prompt, max_new_tokens=30)
        
        print("Generated Text:\n", gen_text)
        summary = summarize_metrics(metrics)
        print("Summary Metrics:", summary)
        # If desired: print per-step breakdown
        for i, m in enumerate(metrics):
            print(f" Step {i+1:02d}: token={m['token']}\tentropy={m['entropy']:.2f}\ttop1={m['top1_prob']:.3f}")
