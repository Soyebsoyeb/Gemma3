"""
Example of model quantization for deployment.
"""
import torch
import torch.nn as nn
from src.models import Gemma3Model, GEMMA3_CONFIG_270M

def quantize_dynamic_example():
    """Example of dynamic quantization."""
    print("=" * 50)
    print("Dynamic Quantization Example")
    print("=" * 50)
    
    # Load model
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load("best_model_params.pt"))
    model.eval()
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8
    )
    
    # Compare sizes
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size / 1e6:.2f} MB")
    print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    # Test inference speed
    import time
    
    dummy_input = torch.randint(0, 1000, (1, 128))
    
    # Original model speed
    start = time.time()
    with torch.no_grad():
        output_original = model(dummy_input)
    original_time = time.time() - start
    
    # Quantized model speed
    start = time.time()
    with torch.no_grad():
        output_quantized = quantized_model(dummy_input)
    quantized_time = time.time() - start
    
    print(f"Original inference time: {original_time:.4f}s")
    print(f"Quantized inference time: {quantized_time:.4f}s")
    print(f"Speedup: {original_time / quantized_time:.2f}x")

def static_quantization_example():
    """Example of static quantization (requires calibration)."""
    print("=" * 50)
    print("Static Quantization Example")
    print("=" * 50)
    
    # Static quantization is more complex and requires calibration
    # This is a placeholder to show the structure
    
    # 1. Prepare model for quantization
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 2. Fuse layers if needed
    # model = torch.quantization.fuse_modules(model, [['layer1', 'layer2']])
    
    # 3. Prepare for calibration
    # model_prepared = torch.quantization.prepare(model)
    
    # 4. Calibrate with representative dataset
    # for batch in calibration_dataloader:
    #     model_prepared(batch)
    
    # 5. Convert to quantized model
    # quantized_model = torch.quantization.convert(model_prepared)

if __name__ == "__main__":
    quantize_dynamic_example()
    # static_quantization_example()
