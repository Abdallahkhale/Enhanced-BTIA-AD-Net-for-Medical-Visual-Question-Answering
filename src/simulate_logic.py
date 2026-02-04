
import torch
import torch.nn as nn
import torch.nn.functional as F

def simulate_logic():
    print("Simulating Embedding Matching Logic...")
    
    # Simulate features
    B, N, D = 4, 10, 768
    projected = torch.randn(B, D)
    embeddings = torch.randn(N, D)
    
    # Normalize
    projected = F.normalize(projected, p=2, dim=-1)
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Sim
    sim = torch.matmul(projected, embeddings.t())
    
    # Temperature
    temp = torch.tensor(0.07)
    
    logits = sim / temp
    probs = F.softmax(logits, dim=-1)
    
    print(f"Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
    print(f"Probs Max: {probs.max().item():.4f}, Min: {probs.min().item():.4f}")
    
    # Simulate high temperature (flat distribution)
    temp_high = torch.tensor(1.0)
    logits_high = sim / temp_high
    probs_high = F.softmax(logits_high, dim=-1)
    print(f"High Temp Probs Max: {probs_high.max().item():.4f}")
    
    # Simulate low temperature (sharp distribution)
    temp_low = torch.tensor(0.01)
    logits_low = sim / temp_low
    probs_low = F.softmax(logits_low, dim=-1)
    print(f"Low Temp Probs Max: {probs_low.max().item():.4f}")

if __name__ == "__main__":
    simulate_logic()
