import torch
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

class Trainer:
    """Handles the training loop for the SLM model."""
    
    def __init__(
        self,
        model,
        dataset_processor,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.dataset_processor = dataset_processor
        self.config = config
        self.device = device
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.max_iters = config.get('max_iters', 150000)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.min_lr = config.get('min_lr', 5e-4)
        self.eval_iters = config.get('eval_iters', 500)
        self.batch_size = config.get('batch_size', 32)
        self.block_size = config.get('block_size', 128)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 32)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            betas=(0.9, 0.95), 
            weight_decay=0.1, 
            eps=1e-9
        )
        
        # Setup learning rate scheduler
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
        scheduler_warmup = LinearLR(self.optimizer, total_iters=self.warmup_steps)
        scheduler_decay = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.max_iters - self.warmup_steps, 
            eta_min=self.min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[scheduler_warmup, scheduler_decay], 
            milestones=[self.warmup_steps]
        )
        
        # Setup mixed precision training
        self.dtype = config.get('dtype', 'bfloat16')
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train(self, save_path: str = "best_model_params.pt"):
        """Run the training loop."""
        from .utils import estimate_loss
        
        for epoch in tqdm(range(self.max_iters)):
            # Evaluate
            if epoch % self.eval_iters == 0 and epoch != 0:
                losses = estimate_loss(
                    self.model, 
                    self.dataset_processor.get_batch,
                    self.eval_iters,
                    self.ctx
                )
                print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                print(f"The current learning rate: {self.optimizer.param_groups[0]['lr']:.5f}")
                
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                
                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    torch.save(self.model.state_dict(), save_path)
            
            # Training step
            X, y = self.dataset_processor.get_batch("train", self.batch_size, self.block_size, self.device)
            
            with self.ctx:
                logits, loss = self.model(X, y)
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            
            # Gradient accumulation and update
            if ((epoch + 1) % self.gradient_accumulation_steps == 0) or (epoch + 1 == self.max_iters):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            self.scheduler.step()
    
    def plot_losses(self, save_path: Optional[str] = None):
        """Plot training and validation losses."""
        import matplotlib.pyplot as plt
        
        train_losses = [i.cpu().detach() if torch.is_tensor(i) else i for i in self.train_losses]
        val_losses = [i.cpu().detach() if torch.is_tensor(i) else i for i in self.val_losses]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'g', label='train_loss')
        plt.plot(val_losses, 'r', label='validation_loss')
        plt.xlabel("Steps - Every 100 epochs")
        plt.ylabel("Loss")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
