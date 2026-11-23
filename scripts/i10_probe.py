"""Paired, fixed-checkpoint tests of the I10 condition on frozen LGC validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "base": ("replicate_base_seed42_v1", "27f34bc6103417a0d3ef19911dbb1d1116b67ebc3b50f288539ca2c7e05c7e8a"),
    "i10_unmixing": ("replicate_i10_unmixing_seed42_v1", "d118d2ad159b1852bf13da61e5e7f7f7dc720730ffa26e99cb7e90bb16d1861a"),
}
MANIFEST = ROOT / "work/configs/screening_2026/manifest_lgc_frozen.csv"
MANIFEST_SHA256 = "2738a2a3ec26388d1bb2887db86d41e3b24ce7b3fecb89578938428a79a60e49"
FROZEN_FILES = {
    "work/tools/train_research_2026.py": "2abb9e37f219bdea7ea8154789b68d3fc985549a40f190784a3ef5b7b0b6f134",
    "work/tools/research_methods_2026.py": "9a24545e72aea07d864d62cf237b3c1fb3b6ef1df4afdc0825e7402b80a1b024",
    "work/tools/benchmark_metrics.py": "10e3e62a9a9c2cd4b4cd7a8abdc6f75944679ac03226ac9cff71e87efd5e3afd",
    "work/repos/STF/src/model/stfdiff/model6/bi_pred_resnet.py": "e628ff755eeddc9e72494c800f54544526680ec78b6ca0dd58c798ce90462363",
}
MODES = ("base", "i10_full", "i10_no_coarse_delta", "i10_shuffle_coarse_delta", "i10_no_injection")
OBSERVATIONS = ("f0", "f3", "c0", "c1", "c3")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_digest(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def evenly_spaced(size: int, count: int) -> list[int]:
    if not 1 <= count <= size:
        raise ValueError(f"Select between 1 and {size} patches")
    if count == 1:
        return [size // 2]
    return [round(i * (size - 1) / (count - 1)) for i in range(count)]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=int, default=24)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_digest(MANIFEST, MANIFEST_SHA256)
    for filename, stamp in FROZEN_FILES.items():
        require_digest(ROOT / filename, stamp)
    out = args.out.resolve()
    run_root = ROOT / "work/runs/research_2026"
    if out.is_relative_to(run_root.resolve()):
        raise ValueError("Use a separate probe output directory")
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")

    sys.path.insert(0, str(ROOT / "work/tools"))
    from train_research_2026 import Patches, ResearchModel, alphas, atomic_json, sample, stable_seed, unpack
    from benchmark_metrics import compute_metrics, from_minus_one_one
    from research_methods_2026 import weights
    import torch
    from torch.utils.data import DataLoader, Subset

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    models = {}
    for method, (directory, stamp) in CHECKPOINTS.items():
        path = run_root / directory / "best.pth"
        require_digest(path, stamp)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["method"], saved["seed"], saved["epoch"]) != (method, 42, 32):
            raise ValueError(f"Unexpected trained weights: {path}")
        model = ResearchModel(method, 42)
        model.load_state_dict(saved["model"], strict=True)
        models[method] = model.eval().to(device)
    base, i10 = models["base"], models["i10_unmixing"]

    dataset = Patches("LGC", "val")
    want = {row["patch_id"] for row in dataset.dates.values()}
    if len(want) != 120 or {path.stem for path in dataset.files} != want:
        raise ValueError("The complete 120-patch LGC development validation split is required")
    selected = evenly_spaced(len(dataset), args.patches)
    for index in selected:
        path = dataset.files[index]
        require_digest(path, dataset.dates[path.stem]["sha256"])
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False)
    alpha = alphas(device)
    t = torch.full((1,), 99, device=device, dtype=torch.long)

    def normalized(target):
        positive = target.clamp_min(1e-6)
        return positive / positive.sum(dim=1, keepdim=True)

    def permute_blocks(delta, key):
        size = 32
        batch, channels, height, width = delta.shape
        if (height % size) or (width % size):
            raise ValueError("Expected spatial axes divisible by 32")
        ny, nx = height // size, width // size
        tiles = delta.reshape(batch, channels, ny, size, nx, size).permute(0, 2, 4, 1, 3, 5)
        tiles = tiles.reshape(batch, ny * nx, channels, size, size)
        generator = torch.Generator().manual_seed(stable_seed(key, 0, "i10_coarse_blocks_v1"))
        order = torch.randperm(ny * nx, generator=generator).to(delta.device)
        shuffled = tiles.index_select(1, order).reshape(batch, ny, nx, channels, size, size)
        return shuffled.permute(0, 3, 1, 4, 2, 5).reshape_as(delta)

    rows, conditions = [], []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            obs, truth = unpack(batch, device)
            key = batch["key"][0]
            left, right = weights(obs)
            a = {name: i10.abundance((obs[name] + 1) / 2).softmax(1) for name in OBSERVATIONS}
            fine = left * a["f0"] + right * a["f3"]
            delta = a["c1"] - left * a["c0"] - right * a["c3"]
            full = i10.prepare(obs)
            target = normalized(fine + 0.3 * delta)
            torch.testing.assert_close(full["feat"], 0.05 * i10.abundance_features(target))
            omit = 0.05 * i10.abundance_features(normalized(fine))
            shuffle = 0.05 * i10.abundance_features(normalized(fine + 0.3 * permute_blocks(delta, key)))
            prepared = (
                (base, base.prepare(obs)), (i10, full), (i10, {**full, "feat": omit}),
                (i10, {**full, "feat": shuffle}), (i10, {**full, "feat": None}),
            )
            maximum, argmax = target.max(dim=1)
            frequency = torch.bincount(argmax.flatten(), minlength=8).float() / argmax.numel()
            conditions.append({
                "key": key, "entropy_nat": float((-(target * target.log()).sum(1)).mean()),
                "mean_maximum_weight": float(maximum.mean()),
                "coarse_delta_rms": float(delta.square().mean().sqrt()),
                "injection_rms": float(full["feat"].square().mean().sqrt()),
                "delta_removal_injection_change_rms": float((full["feat"] - omit).square().mean().sqrt()),
                "shuffle_injection_change_rms": float((full["feat"] - shuffle).square().mean().sqrt()),
                **{f"channel_{j}_occupancy": float(f) for j, f in enumerate(frequency)},
            })
            gt = from_minus_one_one(truth[0].permute(1, 2, 0).cpu().numpy())
            for repeat in range(args.repeats):
                seed = stable_seed(key, repeat)
                rng = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(obs["f0"].shape, generator=rng, device=device)
                for mode, (model, cond) in zip(MODES, prepared, strict=True):
                    pred = model.predict(cond, noise, t).clamp(-1, 1)
                    if index == repeat == 0 and mode in ("base", "i10_full"):
                        torch.testing.assert_close(pred, sample(model, obs, alpha, seed))
                    scores = compute_metrics(from_minus_one_one(pred[0].permute(1, 2, 0).cpu().numpy()),
                                             gt, ergas_ratio=1 / 16)
                    rows.append({
                        "key": key, "target_date": batch["date"][0], "repeat": repeat, "noise_seed": seed,
                        "mode": mode, "rmse": float(((pred - truth) * .5).square().mean().sqrt()),
                        "sam_rad": scores["sam_rad"], "ssim": scores["ssim"],
                        "ergas": scores["ergas"], "cc": scores["cc"],
                    })
            print(f"{index+1}/{len(selected)} {key}", flush=True)

    by_mode = {name: [r for r in rows if r["mode"] == name] for name in MODES}
    reference = by_mode["i10_full"]
    summary = []
    for mode, items in by_mode.items():
        metrics = {}
        for name in ("rmse", "sam_rad", "ssim", "ergas", "cc"):
            finite = [r[name] for r in items if math.isfinite(r[name])]
            metrics[name] = fmean(finite) if finite else None
        metrics["rmse_minus_i10_full"] = fmean(a["rmse"] - b["rmse"] for a, b in zip(items, reference, strict=True))
        summary.append({"mode": mode, **metrics})
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "paired_scores.csv", rows)
    write_rows(out / "conditions.csv", conditions)
    atomic_json(out / "probe_report.json", {
        "scope": "fixed-checkpoint interventions on exposed LGC development validation",
        "training": False, "checkpoint_seed": 42, "checkpoint_epoch": 32, "n_patches": len(selected),
        "noise_repeats": args.repeats, "sampler_steps": 1, "pytorch": torch.__version__, "device": device,
        "manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256": {mode: value[1] for mode, value in CHECKPOINTS.items()},
        "results": summary,
    })
    print(out / "probe_report.json", flush=True)


if __name__ == "__main__":
    main()
