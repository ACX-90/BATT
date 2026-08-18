"""四档增量对照：冻结 / 合集重训 / Replay / 只微调 / 缩放。

新年份默认用电阻整张缩放冒充老化或换对象（同一套短网格 SEQUENCE）。
请在仓库根目录运行。

    python Src/AI/KF/compare.py --make-new --r0-scale 1.15 --r1-scale 1.15
    python Src/AI/KF/compare.py --new-dir Data/soh_k115 --epochs 10
    python Src/AI/KF/compare.py --smoke
    python Src/AI/KF/hole.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
SIM_DIR = KF_DIR.parent.parent / "Sim"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from KF.config import REPO_ROOT
from KF.increment import run_increment


MODES = ("frozen", "retrain", "replay", "finetune", "scale")

MODE_HELP = {
    "frozen": "不更新，冻结权重直接推理（基线）",
    "retrain": "旧网格 + 新年份合集，从旧权重接着训（冻 scaler）",
    "replay": "新年份 + 旧轨迹回放混批",
    "finetune": "只扫新年份，旧温区容易忘",
    "scale": "冻 MLP，只学 k0 k1（整体涨阻正途）",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _count_csv(data_dir: Path) -> int:
    if not data_dir.is_dir():
        return 0
    return sum(1 for p in data_dir.glob("*.csv") if p.name != "index.csv")


def _ensure_new_grid(args: argparse.Namespace) -> Path:
    new_dir = _resolve(args.new_dir)
    if _count_csv(new_dir) > 0 and not args.make_new:
        return new_dir
    if not args.make_new and _count_csv(new_dir) == 0:
        raise FileNotFoundError(
            f"新年份目录为空：{new_dir}  先加 --make-new --r0-scale 1.15 --r1-scale 1.15"
        )
    from nmc100ah_ecm_gen_grid import run_grid

    print(
        f"生成新年份网格 -> {new_dir}  "
        f"R0×{args.r0_scale:g}  R1×{args.r1_scale:g}  C1×{args.c1_scale:g}"
    )
    run_grid(
        n_soc=args.n_soc,
        n_temp=args.n_temp,
        output_dir=new_dir,
        r0_scale=args.r0_scale,
        r1_scale=args.r1_scale,
        c1_scale=args.c1_scale,
    )
    return new_dir


def _ensure_old_grid(args: argparse.Namespace) -> Path:
    old_dir = _resolve(args.old_dir)
    if _count_csv(old_dir) > 0:
        return old_dir
    if not args.make_old:
        raise FileNotFoundError(f"旧网格为空：{old_dir}  先跑 gen_grid，或加 --make-old")
    from nmc100ah_ecm_gen_grid import run_grid

    print(f"生成 BOL 网格 -> {old_dir}")
    run_grid(n_soc=args.n_soc, n_temp=args.n_temp, output_dir=old_dir)
    return old_dir


def _ensure_mlp(args: argparse.Namespace, old_dir: Path) -> Path:
    mlp_dir = _resolve(args.mlp_dir)
    if (mlp_dir / "best.pt").exists() and (mlp_dir / "scaler.json").exists():
        return mlp_dir
    if not args.make_mlp:
        raise FileNotFoundError(
            f"缺少 {mlp_dir / 'best.pt'} 或 scaler.json。先训练，或对照加 --make-mlp"
        )
    from MLP.config import TrainConfig
    from MLP.train import run_training

    print(f"对照用小网训练 -> {mlp_dir}  data={old_dir}")
    cfg = TrainConfig(
        data_dir=str(old_dir.relative_to(REPO_ROOT) if old_dir.is_relative_to(REPO_ROOT) else old_dir),
        out_dir=str(mlp_dir.relative_to(REPO_ROOT) if mlp_dir.is_relative_to(REPO_ROOT) else mlp_dir),
        epochs=args.mlp_epochs,
        pretrain_epochs=2,
        batch_size=4,
    )
    run_training(cfg)
    return mlp_dir


def _incr_ns(overrides: dict) -> argparse.Namespace:
    base = {
        "mode": "replay",
        "mlp_dir": "Data/ai_mlp",
        "out_dir": "Data/ai_kf/incr",
        "ckpt": None,
        "epoch": None,
        "best": True,
        "new_dir": "Data/ai_kf/logs",
        "new_glob": None,
        "new_style": "grid",
        "replay_dir": "Data/grid",
        "replay_glob": None,
        "replay_n": None,
        "beta": 1.0,
        "beta_eval": 1.0,
        "epochs": 10,
        "lr": 5.0e-4,
        "batch_size": None,
        "val_ratio": 0.2,
        "seed": 42,
        "device": None,
        "use_true_inputs": False,
        "eval_only": False,
        "eval_old_dir": None,
        "from_scratch": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def run_one(mode: str, args: argparse.Namespace, old_dir: Path, new_dir: Path) -> dict:
    out_dir = _resolve(args.out_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "mlp_dir": str(_resolve(args.mlp_dir)),
        "out_dir": str(out_dir),
        "new_dir": str(new_dir),
        "eval_old_dir": str(old_dir),
        "new_style": "grid",
        "best": True,
        "ckpt": None,
        "epoch": None,
        "use_true_inputs": args.use_true_inputs,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "from_scratch": False,
        "new_glob": None,
        "replay_glob": None,
    }
    if mode == "frozen":
        ns = _incr_ns(
            {
                **common,
                "mode": "finetune",
                "replay_dir": "",
                "replay_n": 0,
                "epochs": 1,
                "lr": args.lr,
                "beta": 1.0,
                "beta_eval": 1.0,
                "eval_only": True,
            }
        )
    elif mode == "retrain":
        ns = _incr_ns(
            {
                **common,
                "mode": "retrain",
                "replay_dir": str(old_dir),
                "replay_n": None,
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": 1.0,
                "beta_eval": 1.0,
                "eval_only": False,
                "from_scratch": args.retrain_scratch,
            }
        )
    elif mode == "replay":
        ns = _incr_ns(
            {
                **common,
                "mode": "replay",
                "replay_dir": str(old_dir),
                "replay_n": args.replay_n,
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": args.beta,
                "beta_eval": args.beta_eval,
                "eval_only": False,
            }
        )
    elif mode == "finetune":
        ns = _incr_ns(
            {
                **common,
                "mode": "finetune",
                "replay_dir": "",
                "replay_n": 0,
                "epochs": args.epochs,
                "lr": args.lr,
                "beta": 1.0,
                "beta_eval": 1.0,
                "eval_only": False,
            }
        )
    elif mode == "scale":
        ns = _incr_ns(
            {
                **common,
                "mode": "scale",
                "replay_dir": "",
                "replay_n": 0,
                "epochs": args.scale_epochs or args.epochs,
                "lr": args.scale_lr,
                "beta": 1.0,
                "beta_eval": 1.0,
                "eval_only": False,
            }
        )
    else:
        raise ValueError(mode)

    print(f"\n======== {mode}  {MODE_HELP[mode]} ========")
    run_increment(ns)
    meta_path = out_dir / "incr.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mode"] = mode
    meta["note"] = MODE_HELP[mode]
    return meta


def _mv(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{float(val) * 1e3:.2f}"


def _pct(after: list | None, before: list | None, idx: int) -> str:
    if not after or not before or before[idx] == 0:
        return "—"
    return f"{(after[idx] / before[idx] - 1.0) * 100:+.1f}%"


TASK_FOOTERS = {
    "aging": [
        "通过口径（涨阻 / `Doc/04` 任务 A）：",
        "",
        "- 新轨迹开环 RMSE 低于冻结直接推理",
        "- `scale` 的两个 k 齐、朝电阻乘子走；旧集变差是缺维，不是遗忘失败",
        "- `finetune` 电压贴上但两通道不齐，记失败对照",
        "- 表是最后一轮，不是 `best.pt`",
    ],
    "hole": [
        "通过口径（填洞 / `Doc/04` 任务 B，`Doc/03-a` §8）：",
        "",
        "- 冻结：新温区明显差于旧集（外推）",
        "- Replay / 重训：新 RMSE 下降；旧集恶化 < 20%；参考点接近增量前",
        "- 缩放：k ≈ 1，新年份几乎不降（全局乘子补不出低温曲面）",
        "- 只微调：新年份会贴、旧中温应变差（失败对照）",
        "- 正途是 Replay / 重训，不是缩放。表是最后一轮，不是 `best.pt`",
    ],
    "iid": [
        "通过口径（同分布负例 / `Doc/04` 任务 C）：",
        "",
        "- 冻结新年份应已接近旧集，不是涨阻那档的二十多毫伏",
        "- 缩放 k 停在 1 附近；参考点几乎不动",
        "- 微调 / Replay 再削新电压却抬旧集，记成同分布过拟合，不是增量成功",
    ],
    "meas": [
        "通过口径（测量列舰队 / `Doc/04` 任务 D）：",
        "",
        "- 对照任务 A 同一张 ×1.15 表；冻结新/旧相对 A 各抬多少，才是输入域差",
        "- 涨阻结论应仍是缩放；这几个毫伏不该推翻 k 为正途",
        "- 若冻结旧集已到和 A 新年份一个量级，先查列有没有用反",
        "- 表是最后一轮，不是 `best.pt`",
    ],
}


def write_table(rows: list[dict], out_dir: Path, task: str = "aging") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "note",
        "new_rmse_before_mV",
        "new_rmse_after_mV",
        "old_rmse_before_mV",
        "old_rmse_after_mV",
        "old_rmse_change_pct",
        "ref_r0_before_mohm",
        "ref_r0_after_mohm",
        "ref_r0_change_pct",
        "ref_r1_before_mohm",
        "ref_r1_after_mohm",
        "ref_r1_change_pct",
        "k0",
        "k1",
    ]
    table_rows: list[dict] = []
    for meta in rows:
        old_b = meta.get("old_rmse_before")
        old_a = meta.get("old_rmse_after")
        rb = meta.get("ref_before_mohm") or [None, None]
        ra = meta.get("ref_after_mohm") or [None, None]
        rec = {
            "mode": meta.get("mode"),
            "note": meta.get("note", ""),
            "new_rmse_before_mV": None if meta.get("new_rmse_before") is None else meta["new_rmse_before"] * 1e3,
            "new_rmse_after_mV": None if meta.get("new_rmse_after") is None else meta["new_rmse_after"] * 1e3,
            "old_rmse_before_mV": None if old_b is None else old_b * 1e3,
            "old_rmse_after_mV": None if old_a is None else old_a * 1e3,
            "old_rmse_change_pct": None
            if old_b in (None, 0) or old_a is None
            else (old_a / old_b - 1.0) * 100,
            "ref_r0_before_mohm": rb[0],
            "ref_r0_after_mohm": ra[0],
            "ref_r0_change_pct": None if not rb[0] or not ra[0] else (ra[0] / rb[0] - 1.0) * 100,
            "ref_r1_before_mohm": rb[1],
            "ref_r1_after_mohm": ra[1],
            "ref_r1_change_pct": None if not rb[1] or not ra[1] else (ra[1] / rb[1] - 1.0) * 100,
            "k0": meta.get("k0"),
            "k1": meta.get("k1"),
        }
        table_rows.append(rec)

    csv_path = out_dir / "compare.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)

    md_lines = [
        "# 增量四档对照",
        "",
        "| 档 | 新 RMSE 前→后 / mV | 旧 RMSE 前→后 / mV | 旧集变化 | 参考点 R0 | 参考点 R1 | k0 / k1 |",
        "|----|---------------------|---------------------|----------|-----------|-----------|---------|",
    ]
    for meta, rec in zip(rows, table_rows):
        ktxt = "—"
        if rec.get("k0") is not None:
            ktxt = f"{rec['k0']:.3f} / {rec['k1']:.3f}"
        old_chg = "—"
        if rec.get("old_rmse_change_pct") is not None:
            old_chg = f"{rec['old_rmse_change_pct']:+.1f}%"
        md_lines.append(
            f"| `{rec['mode']}` | {_mv(meta.get('new_rmse_before'))} → {_mv(meta.get('new_rmse_after'))} "
            f"| {_mv(meta.get('old_rmse_before'))} → {_mv(meta.get('old_rmse_after'))} "
            f"| {old_chg} "
            f"| {rec['ref_r0_before_mohm']:.4f}→{rec['ref_r0_after_mohm']:.4f} "
            f"({_pct(meta.get('ref_after_mohm'), meta.get('ref_before_mohm'), 0)}) "
            f"| {rec['ref_r1_before_mohm']:.4f}→{rec['ref_r1_after_mohm']:.4f} "
            f"({_pct(meta.get('ref_after_mohm'), meta.get('ref_before_mohm'), 1)}) "
            f"| {ktxt} |"
        )
    md_lines.extend(["", *TASK_FOOTERS.get(task, TASK_FOOTERS["aging"]), ""])
    (out_dir / "compare.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "compare.json").write_text(
        json.dumps({"task": task, "rows": rows, "table": table_rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n对照表  {csv_path}")
    print((out_dir / "compare.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="增量四档对照（冻 scaler，开环电压）")
    p.add_argument("--mlp-dir", default="Data/ai_mlp")
    p.add_argument("--old-dir", default="Data/grid")
    p.add_argument("--new-dir", default="Data/soh_k115")
    p.add_argument("--out-dir", default="Data/ai_kf/compare")
    p.add_argument(
        "--task",
        default="aging",
        choices=tuple(TASK_FOOTERS),
        help="只改对照表验收口径：aging=涨阻，hole=填洞，iid=同分布",
    )
    p.add_argument("--modes", default="frozen,retrain,replay,finetune,scale")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--scale-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--scale-lr", type=float, default=1.0e-2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-eval", type=float, default=1.0)
    p.add_argument("--replay-n", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--retrain-scratch", action="store_true")
    p.add_argument("--make-new", action="store_true", help="按电阻乘子重写 --new-dir")
    p.add_argument("--make-old", action="store_true", help="旧目录为空时生成 BOL 网格")
    p.add_argument("--make-mlp", action="store_true", help="没有权重时在旧网格上先训一版")
    p.add_argument("--mlp-epochs", type=int, default=8)
    p.add_argument("--r0-scale", type=float, default=1.15)
    p.add_argument("--r1-scale", type=float, default=1.15)
    p.add_argument("--c1-scale", type=float, default=1.0)
    p.add_argument("--n-soc", type=int, default=5)
    p.add_argument("--n-temp", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="2×2 网格、1 个 epoch，自备数据与小权重")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> argparse.Namespace:
    args.old_dir = "Data/compare_smoke/old"
    args.new_dir = "Data/compare_smoke/new"
    args.out_dir = "Data/compare_smoke/out"
    args.mlp_dir = "Data/compare_smoke/mlp"
    args.make_old = True
    args.make_new = True
    args.make_mlp = True
    args.n_soc = 2
    args.n_temp = 2
    args.epochs = 1
    args.mlp_epochs = 2
    args.r0_scale = 1.15
    args.r1_scale = 1.15
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        args = apply_smoke(args)
    old_dir = _ensure_old_grid(args)
    new_dir = _ensure_new_grid(args)
    _ensure_mlp(args, old_dir)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"未知档位 {unknown}，可选 {MODES}")

    rows = [run_one(mode, args, old_dir, new_dir) for mode in modes]
    write_table(rows, _resolve(args.out_dir), task=args.task)


if __name__ == "__main__":
    main()
