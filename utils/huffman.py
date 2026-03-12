from __future__ import annotations

import heapq
from collections import Counter


class _Node:
    def __init__(self, freq: int, symbol: int | None = None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_codes(node, prefix: str, table: dict[int, str]) -> None:
    if node.symbol is not None:
        table[node.symbol] = prefix or "0"
        return
    _build_codes(node.left, prefix + "0", table)
    _build_codes(node.right, prefix + "1", table)


def huffman_encode_file(input_path: str, output_path: str) -> float:
    with open(input_path, "rb") as f:
        raw = f.read()

    freq = Counter(raw)
    heap = [_Node(fr, symbol=sym) for sym, fr in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        codes = {heap[0].symbol: "0"}
    else:
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, _Node(a.freq + b.freq, left=a, right=b))
        codes: dict[int, str] = {}
        _build_codes(heap[0], "", codes)

    bitstring = "".join(codes[b] for b in raw)
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding

    out = bytearray([padding])
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i : i + 8], 2))

    with open(output_path, "wb") as f:
        f.write(out)

    return len(out) / max(len(raw), 1)
