"""任务 B：填洞对照。从现成网格挖掉某温区，另训舰队，再跑四档。

禁止对 Data/grid 再 gen_grid，禁止覆盖 Data/ai_mlp / Data/ai_kf/compare。
现役舰队已经见过 −10 °C，不能当填洞起点（那是任务 C）。

    python Src/AI/KF/hole.py
    python Src/AI/KF/hole.py --split-only
    python Src/AI/KF/hole.py --compare-only
    python Src/AI/KF/hole.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

KF_DIR = Path(__file__).resolve().parent
AI_DIR = KF_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from KF.compare import MODES, run_one, write_table
from KF.config import REPO_ROOT


TAG_RE = re.compile(r"(T[+-]\d+)\.csv$", re.IGNORECASE)

PROTECTED = (
    "Data/grid",
    "Data/ai_mlp",
    "Data/ai_kf/compare",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _norm_tag(tag: str) -> str:
    raw = tag.strip()
    if raw.lower().endswith(".csv"):
        raw = raw[:-4]
    up = raw.upper().replace("TM", "")
    if up.startswith("T") and len(up) > 1 and up[1] in "+-":
        return f"T{up[1:]}"
    if up[:1] in "+-":
        return f"T{up}"
    raise ValueError(f"温区标记应写成 T-10 或 -10，收到 {tag!r}")


def temp_tag_of(name: str) -> str | None:
    m = TAG_RE.search(name)
    return f"T{m.group(1)[1:]}" if m else None


def list_traces(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob("*.csv") if p.name.lower() != "index.csv")


def _forbid(path: Path, label: str) -> None:
    rel = _rel(path)
    for item in PROTECTED:
        if rel == item or rel.startswith(item + "/"):
            raise ValueError(f"{label} 不能写到 {item}：{rel}")


def split_grid(
    src_dir: Path,
    old_dir: Path,
    new_dir: Path,
    tag: str,
    *,
    refresh: bool = False,
) -> dict:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"源网格不存在：{src_dir}  先有 Data/grid/，不要对本任务再 gen_grid")
    _forbid(old_dir, "旧集目录")
    _forbid(new_dir, "新年份目录")
    if old_dir.resolve() == src_dir.resolve() or new_dir.resolve() == src_dir.resolve():
        raise ValueError("拆分目标不能就是源网格")

    traces = list_traces(src_dir)
    if not traces:
        raise FileNotFoundError(f"{src_dir} 里没有轨迹 CSV")

    hole = [p for p in traces if temp_tag_of(p.name) == tag]
    keep = [p for p in traces if temp_tag_of(p.name) != tag]
    if not hole:
        seen = sorted({temp_tag_of(p.name) or "?" for p in traces})
        raise FileNotFoundError(f"{src_dir} 没有 {tag} 轨迹。文件里的温区标记：{seen}")
    if not keep:
        raise ValueError(f"{src_dir} 拆完没有旧集（全部都是 {tag}）")

    copied = {
        "old": _copy_group(keep, old_dir, src_dir, refresh),
        "new": _copy_group(hole, new_dir, src_dir, refresh),
    }
    _write_split_index(src_dir / "index.csv", old_dir, [p.name for p in keep])
    _write_split_index(src_dir / "index.csv", new_dir, [p.name for p in hole])

    meta = {
        "task": "B",
        "src": _rel(src_dir),
        "tag": tag,
        "old_dir": _rel(old_dir),
        "new_dir": _rel(new_dir),
        "n_src": len(traces),
        "n_old": len(keep),
        "n_new": len(hole),
        "old_files": [p.name for p in keep],
        "new_files": [p.name for p in hole],
        "copied": copied,
        "note": "复制拆分；源网格未改。各自写了只含本目录文件的 index.csv",
    }
    (old_dir / "split.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (new_dir / "split.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"拆分  {src_dir}  {tag}  -> 旧 {len(keep)} 条 {old_dir}  /  新 {len(hole)} 条 {new_dir}")
    return meta


def _copy_group(files: list[Path], dest: Path, src: Path, refresh: bool) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    want = {p.name for p in files}
    have = {p.name for p in list_traces(dest)}
    if have == want and not refresh:
        return "exists"
    if have and have != want and not refresh:
        raise FileExistsError(
            f"{dest} 已有 {len(have)} 条轨迹，和源网格拆分不一致。"
            f"加 --refresh-split 才重拷（仍不碰 {src}）"
        )
    if refresh:
        for path in list_traces(dest):
            path.unlink()
        idx = dest / "index.csv"
        if idx.exists():
            idx.unlink()
    for src_p in files:
        shutil.copy2(src_p, dest / src_p.name)
    return "copied"


def _write_split_index(src_index: Path, dest: Path, names: list[str]) -> None:
    want = set(names)
    dest_rel = _rel(dest)
    rows: list[dict] = []
    if src_index.exists():
        with src_index.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or ["file", "path"])
            for row in reader:
                fname = row.get("file") or Path(row.get("path") or "").name
                if fname not in want:
                    continue
                row["file"] = fname
                row["path"] = f"{dest_rel}/{fname}"
                rows.append(row)
    else:
        fields = ["file", "path"]
        rows = [{"file": n, "path": f"{dest_rel}/{n}"} for n in names]
    if "path" not in fields:
        fields.append("path")
    if "file" not in fields:
        fields.append("file")
    with (dest / "index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_hole(
    old_dir: Path,
    mlp_dir: Path,
    *,
    epochs: int,
    resume: bool,
    fresh: bool,
    device: str | None,
    seed: int,
) -> Path:
    _forbid(mlp_dir, "洞舰队目录")
    if _rel(mlp_dir) == "Data/ai_mlp":
        raise ValueError("洞舰队不能写到 Data/ai_mlp/。那是见过 −10 °C 的现役舰队，填洞会退化成任务 C")

    best = mlp_dir / "best.pt"
    scaler = mlp_dir / "scaler.json"
    if best.exists() and scaler.exists() and not resume and not fresh:
        print(f"已有洞舰队 {best}，跳过训练。重训加 --fresh-mlp，续训加 --resume-mlp")
        return best

    import torch
    from MLP.config import TrainConfig
    from MLP.train import run_training

    cfg = TrainConfig(
        scheme="B",
        data_dir=_rel(old_dir),
        out_dir=_rel(mlp_dir),
        epochs=epochs,
        seed=seed,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"训洞舰队  data={cfg.data_dir}  out={cfg.out_dir}  epochs={epochs}  resume={resume}")
    out = run_training(cfg, resume=resume and not fresh)
    _warn_undertrained(mlp_dir)
    return out or best


def _warn_undertrained(mlp_dir: Path) -> None:
    hist_path = mlp_dir / "history.json"
    if not hist_path.exists():
        return
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    volts = [r for r in history if r.get("phase") == "voltage"]
    if not volts:
        return
    last = volts[-1]
    val = last.get("val_rmse_v", last.get("train_rmse_v"))
    if val is None:
        return
    mv = float(val) * 1e3
    print(f"洞舰队最后一轮 val RMSE={mv:.2f} mV  （现役全网格舰队旧集开环约 4 mV）")
    if mv > 10.0:
        print(
            "警告：洞舰队还没训到填洞基线。接着训："
            "python Src/AI/KF/hole.py --resume-mlp --skip-compare --mlp-epochs 200"
        )


def _compare_ns(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mlp_dir=str(_resolve(args.mlp_dir)),
        out_dir=str(_resolve(args.out_dir)),
        use_true_inputs=args.use_true_inputs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        scale_lr=args.scale_lr,
        scale_epochs=args.scale_epochs,
        beta=args.beta,
        beta_eval=args.beta_eval,
        replay_n=args.replay_n,
        retrain_scratch=False,
    )


def run_hole_compare(args: argparse.Namespace, old_dir: Path, new_dir: Path) -> list[dict]:
    _forbid(_resolve(args.out_dir), "对照输出")
    mlp_dir = _resolve(args.mlp_dir)
    if not (mlp_dir / "best.pt").exists() or not (mlp_dir / "scaler.json").exists():
        raise FileNotFoundError(f"缺少洞舰队 {mlp_dir / 'best.pt'} 或 scaler.json。先训，不要改用 Data/ai_mlp")
    if _rel(mlp_dir) == "Data/ai_mlp":
        raise ValueError("对照挂了 Data/ai_mlp，这是任务 C 不是填洞。洞舰队应在 Data/ai_mlp_hole")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"未知档位 {unknown}，可选 {MODES}")

    ns = _compare_ns(args)
    rows = [run_one(mode, ns, old_dir, new_dir) for mode in modes]
    write_table(rows, _resolve(args.out_dir), task="hole")
    write_verdict(rows, _resolve(args.out_dir), args)
    return rows


def write_verdict(rows: list[dict], out_dir: Path, args: argparse.Namespace) -> None:
    by_mode = {r.get("mode"): r for r in rows}
    lines = [
        "# 任务 B 填洞读表",
        "",
        f"洞舰队 `{args.mlp_dir}`，旧 `{args.old_dir}`，新 `{args.new_dir}`。",
        "口径：旧 \\((s,T)\\) 的 \\(R\\) 不该变；正途是 Replay / 重训，缩放应几乎不动。",
        "",
    ]
    frozen = by_mode.get("frozen") or {}
    f_new = frozen.get("new_rmse_after")
    f_old = frozen.get("old_rmse_after")
    if f_new is not None and f_old is not None:
        ok = f_new > f_old * 1.3
        lines.append(
            f"- 冻结新 {f_new*1e3:.2f} mV / 旧 {f_old*1e3:.2f} mV："
            + ("新温区更差，外推成立。" if ok else "新温区没有明显差于旧集，先查洞舰队是不是其实见过该温。")
        )

    for mode in ("replay", "retrain"):
        rec = by_mode.get(mode)
        if not rec:
            continue
        old_b, old_a = rec.get("old_rmse_before"), rec.get("old_rmse_after")
        new_b, new_a = rec.get("new_rmse_before"), rec.get("new_rmse_after")
        bits = []
        if old_b and old_a is not None:
            chg = (old_a / old_b - 1.0) * 100
            bits.append(f"旧集 {chg:+.1f}%" + ("（<20%，过）" if chg < 20 else "（≥20%，未过填洞口径）"))
        if new_b and new_a is not None:
            bits.append("新年份下降" if new_a < new_b else "新年份没降")
        rb, ra = rec.get("ref_before_mohm") or [None, None], rec.get("ref_after_mohm") or [None, None]
        if rb[0] and ra[0]:
            bits.append(f"参考点 R0 {(ra[0]/rb[0]-1)*100:+.1f}% / R1 {(ra[1]/rb[1]-1)*100:+.1f}%")
        lines.append(f"- `{mode}`：" + "；".join(bits))

    ft = by_mode.get("finetune")
    if ft and ft.get("old_rmse_before") and ft.get("old_rmse_after"):
        chg = (ft["old_rmse_after"] / ft["old_rmse_before"] - 1.0) * 100
        lines.append(f"- `finetune` 旧集 {chg:+.1f}%（填洞下应变差，这是失败对照）")

    sc = by_mode.get("scale")
    if sc:
        k0, k1 = sc.get("k0"), sc.get("k1")
        new_b, new_a = sc.get("new_rmse_before"), sc.get("new_rmse_after")
        ktxt = "—"
        if k0 is not None:
            near = abs(k0 - 1.0) < 0.05 and abs(k1 - 1.0) < 0.05
            ktxt = f"k0={k0:.3f} k1={k1:.3f}" + ("（≈1，过）" if near else "（离 1 较远，乘子在乱补低温）")
        drop = ""
        if new_b and new_a is not None:
            drop = f"；新年份 {new_b*1e3:.2f}→{new_a*1e3:.2f} mV"
        lines.append(f"- `scale`：{ktxt}{drop}")

    lines.extend(
        [
            "",
            "表里是最后一轮。各档 `history.json` 里打 `*` 的才是 `best.pt`。",
            "洞舰队 val RMSE 若仍明显高于 4 mV，先 `--resume-mlp` 再跑对照。",
            "",
        ]
    )
    text = "\n".join(lines)
    (out_dir / "verdict.md").write_text(text, encoding="utf-8")
    print("\n" + text)


def apply_smoke(args: argparse.Namespace) -> argparse.Namespace:
    src = _resolve("Data/compare_smoke/old")
    if not list_traces(src):
        src = _resolve("Data/grid")
    args.src_dir = _rel(src)
    args.old_dir = "Data/compare_smoke/hole/wo"
    args.new_dir = "Data/compare_smoke/hole/tm10"
    args.mlp_dir = "Data/compare_smoke/hole/mlp"
    args.out_dir = "Data/compare_smoke/hole/out"
    args.mlp_epochs = 2
    args.epochs = 1
    args.replay_n = 2
    args.fresh_mlp = True
    args.refresh_split = True
    print("烟测：权重未训稳，数字作废")
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="任务 B：挖掉温区后的四档填洞对照")
    p.add_argument("--src-dir", default="Data/grid", help="现成全网格，只读复制")
    p.add_argument("--old-dir", default="Data/grid_wo_tm10")
    p.add_argument("--new-dir", default="Data/grid_tm10")
    p.add_argument("--mlp-dir", default="Data/ai_mlp_hole")
    p.add_argument("--out-dir", default="Data/ai_kf/compare_hole")
    p.add_argument("--temp-tag", default="T-10", help="挖掉的温区文件标记，如 T-10")
    p.add_argument("--modes", default="frozen,retrain,replay,finetune,scale")
    p.add_argument("--mlp-epochs", type=int, default=100, help="洞舰队电压 epoch；不够再 --resume-mlp")
    p.add_argument("--epochs", type=int, default=10, help="对照各档增量 epoch")
    p.add_argument("--scale-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--scale-lr", type=float, default=1.0e-2)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-eval", type=float, default=1.0)
    p.add_argument("--replay-n", type=int, default=50)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--use-true-inputs", action="store_true")
    p.add_argument("--refresh-split", action="store_true", help="重拷拆分目录，仍不碰源网格")
    p.add_argument("--fresh-mlp", action="store_true", help="洞舰队从头训")
    p.add_argument("--resume-mlp", action="store_true", help="洞舰队从最新 epoch 续训")
    p.add_argument("--split-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--compare-only", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-compare", action="store_true")
    p.add_argument("--smoke", action="store_true", help="小目录 1 个 epoch，数字作废")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args = apply_smoke(args)

    tag = _norm_tag(args.temp_tag)
    src_dir = _resolve(args.src_dir)
    old_dir = _resolve(args.old_dir)
    new_dir = _resolve(args.new_dir)
    mlp_dir = _resolve(args.mlp_dir)

    do_split = not args.compare_only
    do_train = not args.split_only and not args.compare_only and not args.skip_train
    do_compare = not args.split_only and not args.train_only and not args.skip_compare
    if args.train_only:
        do_train = True

    if do_split:
        split_grid(src_dir, old_dir, new_dir, tag, refresh=args.refresh_split)
    else:
        if not list_traces(old_dir) or not list_traces(new_dir):
            raise FileNotFoundError(f"拆分目录是空的：{old_dir} / {new_dir}。先跑不带 --compare-only")

    if do_train:
        train_hole(
            old_dir,
            mlp_dir,
            epochs=args.mlp_epochs,
            resume=args.resume_mlp,
            fresh=args.fresh_mlp,
            device=args.device,
            seed=args.seed,
        )

    if do_compare:
        run_hole_compare(args, old_dir, new_dir)


if __name__ == "__main__":
    main()
