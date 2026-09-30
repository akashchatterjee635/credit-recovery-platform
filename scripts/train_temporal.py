"""Train the temporal/static fusion model from persisted tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import set_random_seeds, write_run_manifest


def _config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def train(config: dict) -> Path:
    try:
        import torch
        from backend.models.deep.fusion import TemporalStaticFusionModel
    except ImportError as exc:
        raise RuntimeError("Install the 'deep' dependency group to train temporal models") from exc

    seed = int(config.get("seed", 42))
    set_random_seeds(seed)
    tensor_dir = Path(config.get("tensor_dir", "data/tensors"))
    metadata_path = Path(config.get("metadata", "artifacts/temporal_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = {
        name: np.load(tensor_dir / f"train_{name}.npy")
        for name in ("X_seq", "mask", "X_cont", "X_cat", "y")
    }
    model = TemporalStaticFusionModel(
        temporal_dim=metadata["temporal_features"],
        static_dim=int(config.get("hidden_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        temporal_model_type=config.get("temporal_model_type", "TCN"),
        ft_params={
            "num_continuous": metadata["num_continuous"],
            "cat_cardinalities": metadata["cat_cardinalities"],
            "d_model": int(config.get("hidden_dim", 64)),
            "nhead": int(config.get("nhead", 4)),
            "num_layers": int(config.get("num_layers", 2)),
            "dropout": float(config.get("dropout", 0.1)),
        },
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensors = {
        "seq": torch.tensor(arrays["X_seq"], dtype=torch.float32),
        "mask": torch.tensor(arrays["mask"], dtype=torch.bool),
        "cont": torch.tensor(arrays["X_cont"], dtype=torch.float32),
        "cat": torch.tensor(arrays["X_cat"], dtype=torch.long),
        "y": torch.tensor(arrays["y"], dtype=torch.float32),
    }
    model.train()
    loss_value = 0.0
    for _ in range(int(config.get("epochs", 1))):
        optimizer.zero_grad()
        logits, _ = model(
            tensors["seq"], tensors["cont"], tensors["cat"], tensors["mask"]
        )
        loss = loss_fn(logits.squeeze(1), tensors["y"])
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())

    output = Path(config.get("output", "artifacts/deep_fusion_model.pth"))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    write_run_manifest(
        command="python -m scripts.train_temporal",
        config=config,
        dataset_paths=[tensor_dir / "train_ids.npy"],
        output_paths=[output, metadata_path],
        feature_order=(
            metadata["continuous_features"]
            + metadata["categorical_features"]
            + metadata["temporal_feature_names"]
        ),
        metrics={"final_training_loss": loss_value},
        seed=seed,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"Saved temporal model to {train(_config(args.config))}")


if __name__ == "__main__":
    main()
