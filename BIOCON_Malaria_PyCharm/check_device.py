"""Check PyTorch CPU / Apple MPS / NVIDIA CUDA availability."""
import sys
import torch

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available :", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Selected device: Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Selected device: NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("Selected device: CPU")

# Small operation test on the selected device
try:
    x = torch.randn(2, 3, 64, 64, device=device)
    conv = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
    y = conv(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    print("Tensor operation test: PASSED")
    print("Output shape:", tuple(y.shape))
except Exception as exc:
    print("Device test failed:", repr(exc))
