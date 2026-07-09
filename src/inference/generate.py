import torch
import torch.nn.functional as F
from typing import Optional, List, Union
from src.models import Gemma3Model
from src.data import Tokenizer

class TextGenerator:
    """Text generation handler for trained SLM models."""
    
    def __init__(
        self,
        model: Gemma3Model,
        tokenizer: Optional[Tokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 32768
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer or Tokenizer()
        self.device = device
        self.max_length = max_length
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt as string or token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated text string
        """
        # Tokenize prompt if string
        if isinstance(prompt, str):
            input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        else:
            input_ids = torch.tensor(prompt).unsqueeze(0).to(self.device)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Check max length
            if input_ids.size(1) >= self.max_length:
                break
            
            # Get logits for next token
            logits, _ = self.model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in input_ids[0]:
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.tensor(float('-inf')).to(next_token_logits.device),
                    next_token_logits
                )
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token generated (assuming tokenizer has EOS)
            if hasattr(self.tokenizer.enc, 'eot_token') and next_token.item() == self.tokenizer.enc.eot_token:
                break
        
        # Decode generated tokens
        return self.tokenizer.decode(input_ids.squeeze().tolist())
    
    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        batch_size: int = 8
    ) -> List[str]:
        """Generate text for multiple prompts in batches."""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                generated = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                batch_results.append(generated)
            
            results.extend(batch_results)
        
        return results

def generate_text(
    model_path: str,
    prompt: str,
    config=None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Convenience function for quick text generation.
    
    Args:
        model_path: Path to model checkpoint
        prompt: Input prompt
        config: Model configuration (defaults to GEMMA3_CONFIG_270M)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run inference on
    
    Returns:
        Generated text
    """
    from src.models import Gemma3Model, GEMMA3_CONFIG_270M
    
    if config is None:
        config = GEMMA3_CONFIG_270M
    
    # Load model
    model = Gemma3Model(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Initialize generator
    generator = TextGenerator(model, device=device)
    
    # Generate
    return generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
