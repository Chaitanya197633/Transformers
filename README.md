# Deep Compression Pipeline (Feature-based MLP)

This repository implements a full deep compression workflow for an MLP classifier trained on image-derived feature vectors.

## Scope constraints
- No CNN is trained.
- Images are used only to derive fixed-size feature vectors.
- Compression is applied only to MLP linear layers.

## Run
```bash
python main.py
```

## Pipeline
1. Build feature vectors from images.
2. Train baseline MLP.
3. Replace linear layers with compression-aware linear layers.
4. Apply magnitude pruning and fine-tune.
5. Quantize pruned weights.
6. Serialize compressed model to NPZ.
7. Huffman-encode NPZ and report compression ratio.
