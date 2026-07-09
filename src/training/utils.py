import torch
import numpy as np

def estimate_loss(model, get_batch_fn, eval_iters, ctx):
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch_fn(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    
    model.train()
    return out
