"""Center-embedding cognitive load project.

This script measures token-level surprisal / cross-entropy as a language model reads
center-embedded sentences with increasing syntactic depth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - dependency/runtime guard
    raise RuntimeError(
        "transformers is required for this project. Install with `pip install transformers`."
    ) from exc


NOUNS = [
    "reporter",
    "senator",
    "lawyer",
    "scientist",
    "pilot",
    "artist",
    "teacher",
]

VERBS_TRANSITIVE = [
    "interviewed",
    "consulted",
    "challenged",
    "thanked",
    "criticized",
    "supported",
    "admired",
]

COMPLEMENTIZER = "that"


@dataclass(frozen=True)
class SentenceSpec:
    depth: int
    sentence: str


def make_center_embedded_sentence(depth: int) -> SentenceSpec:
    """Create a deterministic center-embedded sentence with `depth` relative clauses.

    Example depth=3:
    The reporter that the senator that the lawyer consulted interviewed challenged slept.
    """

    if depth < 1:
        raise ValueError("depth must be >= 1")

    needed = depth + 1
    if needed > len(NOUNS) or depth > len(VERBS_TRANSITIVE):
        raise ValueError(
            f"depth={depth} exceeds template capacity; max depth is {min(len(VERBS_TRANSITIVE), len(NOUNS) - 1)}"
        )

    nouns = NOUNS[: needed]
    verbs = VERBS_TRANSITIVE[:depth]

    sentence_parts: list[str] = ["The", nouns[0]]

    # Open relative clauses: that the N1 that the N2 ...
    for i in range(1, depth + 1):
        sentence_parts.extend([COMPLEMENTIZER, "the", nouns[i]])

    # Close clauses with verbs in LIFO order.
    for i in range(depth - 1, -1, -1):
        sentence_parts.append(verbs[i])

    sentence_parts.append("slept")
    sentence = " ".join(sentence_parts) + "."
    return SentenceSpec(depth=depth, sentence=sentence)


def iter_specs(min_depth: int, max_depth: int) -> Iterable[SentenceSpec]:
    for depth in range(min_depth, max_depth + 1):
        yield make_center_embedded_sentence(depth)


def token_cross_entropy(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token cross-entropy for predicting token[t] from prefix[:t]."""

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
    return losses.view(input_ids.size(0), -1)


def evaluate_sentence(model, tokenizer, sentence: str) -> list[dict]:
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"]

    losses = token_cross_entropy(model, input_ids)[0]
    target_tokens = input_ids[0, 1:]

    decoded = [tokenizer.decode([tok]).strip() for tok in target_tokens]
    records: list[dict] = []
    for idx, (tok, loss) in enumerate(zip(decoded, losses.tolist()), start=1):
        records.append(
            {
                "position": idx,
                "token": tok if tok else "<space>",
                "cross_entropy_nats": loss,
                "surprisal_bits": loss / torch.log(torch.tensor(2.0)).item(),
                "perplexity": float(torch.exp(torch.tensor(loss)).item()),
            }
        )
    return records


def run_experiment(model_name: str, min_depth: int, max_depth: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    all_rows: list[dict] = []

    for spec in iter_specs(min_depth, max_depth):
        rows = evaluate_sentence(model, tokenizer, spec.sentence)
        for row in rows:
            row["depth"] = spec.depth
            row["sentence"] = spec.sentence
            all_rows.append(row)

    # Save as TSV to avoid pandas dependency.
    tsv_path = out_dir / "token_entropy.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "depth\tsentence\tposition\ttoken\tcross_entropy_nats\tsurprisal_bits\tperplexity\n"
        )
        for row in all_rows:
            f.write(
                f"{row['depth']}\t{row['sentence']}\t{row['position']}\t{row['token']}\t"
                f"{row['cross_entropy_nats']:.6f}\t{row['surprisal_bits']:.6f}\t{row['perplexity']:.6f}\n"
            )

    # Plot mean surprisal by token position for each depth.
    plt.figure(figsize=(10, 5))
    for depth in range(min_depth, max_depth + 1):
        depth_rows = [r for r in all_rows if r["depth"] == depth]
        xs = [r["position"] for r in depth_rows]
        ys = [r["surprisal_bits"] for r in depth_rows]
        plt.plot(xs, ys, marker="o", label=f"depth={depth}")

    plt.title(f"Token surprisal curve on center-embedded sentences ({model_name})")
    plt.xlabel("Token position")
    plt.ylabel("Surprisal (bits)")
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / "entropy_curve.png"
    plt.savefig(png_path, dpi=150)

    # Also output a compact summary: mean and peak surprisal by depth.
    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("depth\tmean_surprisal_bits\tpeak_surprisal_bits\n")
        for depth in range(min_depth, max_depth + 1):
            depth_rows = [r for r in all_rows if r["depth"] == depth]
            ys = [r["surprisal_bits"] for r in depth_rows]
            mean_y = sum(ys) / len(ys)
            peak_y = max(ys)
            f.write(f"{depth}\t{mean_y:.6f}\t{peak_y:.6f}\n")

    print(f"Wrote token-level entropy table: {tsv_path}")
    print(f"Wrote depth summary table: {summary_path}")
    print(f"Wrote entropy curve figure: {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure token-level cross-entropy for center-embedded sentences."
    )
    parser.add_argument("--model", default="distilgpt2", help="HF causal LM name")
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/center_embedding"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.model, args.min_depth, args.max_depth, args.out_dir)
