# Cudakit

Cudakit is a minimal PyTorch C++/CUDA extension project with a small set of example ops.
It is intended as a starting point for building and testing custom operators.

## Install

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

## Usage

```python
import torch
from cudakit import ops

a = torch.randn(4, device="cuda")
b = torch.randn(4, device="cuda")

# Fused multiply-add
out = ops.mymuladd(a, b, 1.0)

# Out variant
out2 = torch.empty_like(a)
ops.myadd_out(a, b, out2)
```

## Tests

```bash
pytest test/ -v
```
