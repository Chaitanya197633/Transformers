# Deep Compression Pipeline (Feature-based MLP)

This repository implements a deep compression workflow for an MLP classifier trained on image-derived feature vectors.

## Scope constraints
- No CNN is trained.
- Images are used only to derive fixed-size feature vectors.
- Compression is applied only to MLP linear layers.

## Environment setup (VS Code / Jupyter)
Install dependencies in the **same** interpreter/kernel where you'll run the notebook/script:

```bash
python -m pip install -r requirements.txt
```

In notebooks:

```python
%pip install -r requirements.txt
```

Then restart the kernel/interpreter.

## Windows PyTorch DLL error support
If your environment throws errors like `WinError 1114` / `Error loading ... torch\lib\c10.dll`,
`run_pipeline()` now automatically falls back to a NumPy implementation so you can continue running the full pruning/quantization/NPZ/Huffman flow.

The returned metrics include:
- `backend`: `"torch"` or `"numpy-fallback"`
- `fallback_reason`: included when fallback is used

## Run
```bash
python main.py
```

Notebook:

```python
from main import run_pipeline
metrics = run_pipeline()
metrics
```

## Pipeline
1. Build feature vectors from images.
2. Train baseline MLP.
3. Apply magnitude pruning and fine-tune.
4. Quantize pruned weights.
5. Serialize compressed model to NPZ.
6. Huffman-encode NPZ and report compression ratio.
