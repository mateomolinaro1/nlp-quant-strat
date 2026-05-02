import sys

print(f"Python : {sys.version}")

try:
    import torch
    print(f"torch  : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version   : {torch.version.cuda}")
        print(f"Device count   : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name} — {props.total_memory / 1e9:.1f} GB VRAM")
        t = torch.tensor([1.0]).cuda()
        print(f"Tensor on GPU  : OK ({t.device})")
    else:
        print("Reason         :", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "no CUDA device found")
except ImportError as e:
    print(f"torch not installed: {e}")

try:
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    print(f"\nsentence-transformers : {sentence_transformers.__version__}")
    print("Quick encode test (CPU)...")
    m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    v = m.encode(["hello world"])
    print(f"Embedding shape : {v.shape} — OK")
    if torch.cuda.is_available():
        print("Quick encode test (GPU)...")
        m_gpu = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        v_gpu = m_gpu.encode(["hello world"])
        print(f"Embedding shape : {v_gpu.shape} — OK on GPU")
except ImportError as e:
    print(f"\nsentence-transformers not installed: {e}")
except Exception as e:
    print(f"\nsentence-transformers error: {e}")