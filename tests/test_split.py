from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_nuring import split_indices_by_sample


class DummyDataset:
    records = [type("R", (), {"name": "a"})(), type("R", (), {"name": "b"})(), type("R", (), {"name": "c"})()]
    index = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]

    def __len__(self) -> int:
        return len(self.index)


def main() -> None:
    train_idx, val_idx, meta = split_indices_by_sample(DummyDataset(), {"mode": "image", "val_ratio": 0.34, "seed": 1})
    train_samples = set(meta["train_samples"])
    val_samples = set(meta["val_samples"])
    assert not train_samples & val_samples
    assert len(train_idx) + len(val_idx) == len(DummyDataset().index)
    print("split ok", meta)


if __name__ == "__main__":
    main()
