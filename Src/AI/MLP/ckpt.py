"""按 epoch 编号保存 / 查找权重。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import REPO_ROOT


def ckpt_dir(out_dir: Path) -> Path:
    return out_dir / "ckpts"


def epoch_path(out_dir: Path, epoch: int) -> Path:
    return ckpt_dir(out_dir) / f"epoch_{int(epoch):05d}.pt"


def list_epochs(out_dir: Path) -> list[tuple[int, Path]]:
    folder = ckpt_dir(out_dir)
    items: list[tuple[int, Path]] = []
    if not folder.exists():
        return items
    for path in folder.glob("epoch_*.pt"):
        try:
            num = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        items.append((num, path))
    items.sort()
    return items


def latest_epoch(out_dir: Path) -> tuple[int, Path] | None:
    items = list_epochs(out_dir)
    return items[-1] if items else None


def write_latest_pointer(out_dir: Path, epoch: int) -> None:
    folder = ckpt_dir(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": int(epoch), "file": epoch_path(out_dir, epoch).name}
    (folder / "latest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_ckpt_path(
    out_dir: Path,
    *,
    epoch: int | None = None,
    ckpt: str | None = None,
    fallback_last: Path | None = None,
    fallback_best: Path | None = None,
) -> Path:
    if ckpt:
        path = Path(ckpt)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"找不到权重: {path}")
        return path
    if epoch is not None:
        path = epoch_path(out_dir, epoch)
        if not path.exists():
            available = ", ".join(str(n) for n, _ in list_epochs(out_dir)) or "无"
            raise FileNotFoundError(f"没有 epoch {epoch} 的权重 ({path.name})。已有: {available}")
        return path
    latest = latest_epoch(out_dir)
    if latest is not None:
        return latest[1]
    if fallback_last is not None and fallback_last.exists():
        return fallback_last
    if fallback_best is not None and fallback_best.exists():
        return fallback_best
    raise FileNotFoundError(f"{out_dir} 下没有 epoch_*.pt / last.pt / best.pt")


def format_epoch_list(out_dir: Path) -> str:
    items = list_epochs(out_dir)
    if not items:
        return "尚无按 epoch 保存的权重"
    return "已保存 epoch: " + ", ".join(str(n) for n, _ in items) + f"  (最新 {items[-1][0]})"
