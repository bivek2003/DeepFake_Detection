"""
Audio testing utilities with memory management
"""

import torch
import gc

def safe_audio_test(model, max_retries=3, start_size=16000):
    """Memory-safe audio testing with progressive sizing"""
    model.eval()
    sizes_to_try = [start_size, start_size*2, start_size*3, start_size*4]
    
    for size in sizes_to_try:
        for attempt in range(max_retries):
            try:
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create smaller batch size for memory efficiency
                batch_size = 1 if size > 32000 else 2
                
                with torch.no_grad():
                    audio_input = torch.randn(batch_size, size)
                    output = model(audio_input)
                    
                    print(f"✅ Audio test successful: {audio_input.shape} -> {output.shape}")
                    return True
                    
            except RuntimeError as e:
                if "memory" in str(e).lower() or "alloc" in str(e).lower():
                    print(f"   Memory issue with size {size}, attempt {attempt+1}")
                    continue
                else:
                    print(f"   Non-memory error: {str(e)[:50]}...")
                    break
            except Exception as e:
                print(f"   Error with size {size}: {str(e)[:50]}...")
                break
    
    print("❌ All audio test attempts failed")
    return False

def optimize_memory_usage():
    """Optimize memory usage for audio processing"""
    import torch
    import gc
    
    # Clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set memory efficient settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print("✅ Memory usage optimized")
