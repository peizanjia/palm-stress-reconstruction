"""Freeze height-to-force estimators, then score a fresh blind FEM batch."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F

from train_surface_surrogate import MESH, ROOT, VariationalModel, sync


PROTOCOL = ROOT/"config/generalization_blind.json"
METHODS = ("ridge", "image", "pinn")
HISTORY_GROUPS = {"train", "development"}
BLIND_PREFIXES = {"test": "blind_test", "combined": "blind_combined"}
RESULT_COLUMNS = ("Trajectory", "Travel_mm", "Reference_N", "Estimate_N", "Latency_ms",
                  "MinJGauss", "FastForceDifference_N")


class Observations:
    def __init__(self, device):
        self.device = device
        axis = torch.linspace(-25, 25, 201, device=device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.xy = torch.stack((x, y), dim=-1)
        self.blur = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=torch.float32)/16
        p = torch.linspace(-7, 7, 29, device=device)
        v, u = torch.meshgrid(p, p, indexing="ij")
        self.patch = torch.stack((u, v), dim=-1)
        self.levels = torch.tensor([0.2, 0.5, 0.8], device=device)[None, :, None, None]
        self.moments = torch.stack((torch.ones_like(u), u.square(), v.square(), u*v), dim=-1)

    def __call__(self, raw, physical=False):
        kernel = self.blur
        z = F.conv2d(raw[:, None], kernel[None, None, None, :], padding=(0, 2))
        z = F.conv2d(z, kernel[None, None, :, None], padding=(2, 0))
        depth = z.amax((-1, -2, -3)).clamp(min=0)
        scale = depth.clamp(min=0.03)
        weight = (z[:, 0]-0.85*scale[:, None, None]).clamp(min=0)
        center = torch.einsum("bij,ijd->bd", weight, self.xy)/weight.sum((1, 2)).clamp(min=1e-7)[:, None]
        grid = (self.patch[None]+center[:, None, None, :])/25
        patch = F.grid_sample(z, grid, align_corners=True)/scale[:, None, None, None]
        footprint = patch[:, 0].clamp(0, 1)
        average = footprint.mean((1, 2))
        weights = ((footprint[:, None]-self.levels)/0.12).clamp(0, 1)
        total, xx, yy, xy = torch.einsum("blij,ijd->bld", weights, self.moments).unbind(-1)
        total = total.clamp(min=1e-7)
        xx, yy, xy = xx/total, yy/total, xy/total
        anisotropy = torch.sqrt((xx-yy).square()+4*xy.square()+1e-8)
        moments = torch.stack((total/50, (xx+yy)/16, anisotropy/(xx+yy+0.05)), dim=-1)
        features = torch.cat((torch.stack((depth/3, torch.log1p(8*depth)/4,
                                           center.norm(dim=1)/10, average*5), dim=1),
                              moments.flatten(1)), dim=1)
        canonical = None
        if physical:
            canonical = F.grid_sample(z, (self.xy[None]+center[:, None, None, :])/25,
                                      align_corners=True)[:, 0]
        return patch, features, depth, canonical


class Decoder(VariationalModel):
    """Anisotropic trial functions with the observed top graph built in."""

    def __init__(self, device):
        super().__init__(device)
        xy = self.nodes[:, :2]
        x, y = xy.unbind(dim=1)
        r2 = x.square()+y.square()
        g = torch.exp(-r2[:, None]/self.tensor([8, 32, 98]))*self.rim[:, None]
        z = torch.zeros_like(x)
        radial = xy[:, None, :]/self.tensor([2, 4, 7])[None, :, None]*g[..., None]
        angular = torch.stack((torch.stack((x, -y), 1)/4, torch.stack((y, x), 1)/4,
                               torch.stack((1+z, z), 1), torch.stack((z, 1+z), 1)), dim=1)
        self.in_plane = torch.cat((radial, angular*g[:, 1, None, None]), dim=1)
        self.vertical = torch.cat((g, g[:, 1:2]*torch.stack(
            (x/4, y/4, (x.square()-y.square())/16, x*y/8), dim=1)), dim=1)

    def displacement(self, heights, depth, coeff):
        a = torch.tanh(coeff)*depth[:, None]
        t = self.fraction[None, :, None]
        top = torch.einsum("nki,bk->bni", self.in_plane, a[:, :7])
        flow = torch.einsum("nki,bk->bni", self.in_plane, a[:, 7:14])
        xy = t*(top+(1-t)*flow)
        positions = (self.nodes[None, :, :2]+top)/25
        observed = F.grid_sample(heights[:, None], positions[:, None], mode="bicubic",
                                 align_corners=True)[:, 0, 0]
        dz = torch.einsum("nk,bk->bn", self.vertical, a[:, 14:21])
        curvature = torch.einsum("nk,bk->bn", self.vertical, a[:, 21:28])
        dz = dz+(2*t[..., 0]-1)*curvature
        vertical = -t[..., 0]*observed*self.active_node+t[..., 0]*(1-t[..., 0])*dz
        return torch.cat((xy, vertical[..., None]), dim=2)

    def decode(self, heights, depth, coeff, force_only=False):
        return self.evaluate_displacement(self.displacement(heights, depth, coeff), force_only)


class Network(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(1, 12, 5, stride=2, padding=2), nn.SiLU(),
                                   nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.SiLU(),
                                   nn.AdaptiveAvgPool2d(3), nn.Flatten())
        self.out = nn.Sequential(nn.Linear(144+13, 48), nn.SiLU(), nn.Linear(48, outputs))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, patch, features):
        return self.out(torch.cat((self.image(patch), features), dim=1))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_hashes():
    files = [PROTOCOL, MESH, Path(__file__).resolve(), ROOT/"tools/train_surface_surrogate.py",
             ROOT/"run_generalization_samples.m", ROOT/"run_indentation.m",
             ROOT/"run_blind_surface_inverse.m"]
    files += sorted((ROOT/"config").glob("*.m"))
    files += sorted((ROOT/"src").rglob("*.m"))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}


def tool_pose(spec):
    axes = np.asarray(spec["axes"]).tolist()
    sphere = spec["power"] == 2 and np.ptp(axes) < 1e-10
    angles = [0, 0] if sphere else [spec["yaw_deg"], spec["tilt_deg"]]
    return [spec["power"], *axes, *angles, *np.asarray(spec["center_mm"]).tolist()]


def read_dataset(folders, device, protocol, freeze_token=None):
    groups = {}
    records = []
    for folder in folders:
        if not folder.is_dir():
            raise ValueError(f"Sample directory does not exist: {folder}")
    paths = sorted(path for folder in folders for path in folder.glob("*.mat")
                   if path.stem != "design")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Trajectory identifiers must be unique across data directories.")
    blind = freeze_token is not None
    if blind:
        expected = {
            f"{prefix}_{i:02d}": (group, protocol[f"{kind}_nonzero_steps"])
            for group, prefix in BLIND_PREFIXES.items()
            for kind in ("regular" if group == "test" else group,)
            for i in range(1, protocol[f"{kind}_trajectories"]+1)
        }
        if {path.stem for path in paths} != expected.keys():
            raise ValueError("Blind batch differs from the prescribed full trajectory list.")
    elif any(path.stem.startswith("blind_") for path in paths):
        raise ValueError("Final blind samples cannot be supplied to training.")

    for path in paths:
        if blind:
            group, steps = expected[path.stem]
        else:
            group = path.stem.split("_")[0]
            if group not in HISTORY_GROUPS:
                continue
            spec = loadmat(path, simplify_cells=True, variable_names=["spec"])["spec"]
            if 60 <= spec["yaw_deg"] <= 100 and not (
                    spec["power"] == 2 and np.ptp(spec["axes"]) < 1e-10):
                continue
            steps = len(spec["depths_mm"])
        mat = loadmat(path, simplify_cells=True, variable_names=[
            "heights", "travel", "force", "spec", "diagnostics", "seconds", "freezeToken"])
        spec = mat["spec"]
        if (blind and (spec["id"] != path.stem or mat.get("freezeToken") != freeze_token) or
                not blind and mat.get("freezeToken") is not None):
            raise ValueError(f"{path.name} has mismatched identity or was not generated after this freeze.")
        travel = np.atleast_1d(mat["travel"])
        force = np.atleast_1d(mat["force"])
        diagnostics = np.atleast_2d(mat["diagnostics"])
        if (len(travel) != steps+1 or
                not np.array_equal(travel, np.r_[0, spec["depths_mm"]]) or
                not np.isfinite(diagnostics[:, 2:]).all() or not np.isfinite(force).all() or
                np.any(abs(force-diagnostics[:, 4]) >= 1e-6) or
                np.any(force[1:] <= 0) or np.any(diagnostics[:, 6] <= 0) or
                np.any(diagnostics[:, 7] > protocol["reference_residual_limit_N"])):
            raise ValueError(f"{path.name} failed the complete-trajectory reference checks.")
        # Score every prescribed nonzero step. A bad reference aborts the batch.
        array = np.moveaxis(mat["heights"], -1, 0)[1:]
        valid = np.isfinite(array)
        if array.shape != (steps, 201, 201) or not valid.any(axis=(1, 2)).all():
            raise ValueError(f"{path.name} has missing or malformed height frames.")
        raw = np.where(valid, 5-array, 0)
        frame_hashes = [hashlib.sha256(np.ascontiguousarray(frame, dtype="<f4")).hexdigest()
                        for frame in raw]
        records.append(dict(trajectory=path.stem, group=group, frames=steps,
                            seconds=mat["seconds"], input_sha256=sha256(path),
                            tool_pose=tool_pose(spec), frame_sha256=frame_hashes))
        groups.setdefault(group, []).append((raw, force[1:], travel[1:], valid,
                                            np.repeat(path.stem, steps)))
    joined = {}
    for group, rows in groups.items():
        z, forces, depth, mask, labels = (np.concatenate(field) for field in zip(*rows))
        joined[group] = (torch.as_tensor(z, device=device, dtype=torch.float32),
                         torch.as_tensor(forces, device=device, dtype=torch.float32),
                         depth, mask, labels)
    return joined, records


def check_disjoint(first, second):
    for field, items in (
        ("IDs", lambda row: [row["trajectory"]]),
        ("tool geometries and poses", lambda row: [tuple(row["tool_pose"])]),
        ("height frames", lambda row: row["frame_sha256"]),
    ):
        if {item for row in first for item in items(row)} & {item for row in second for item in items(row)}:
            raise ValueError(f"Dataset splits share {field}.")


def predict(network, inputs, decoder=None, force_only=False):
    patch, scalars, depth, height = inputs
    coeff = network(patch, scalars)
    if decoder is not None:
        energy, force, minimum = decoder.decode(height, depth, coeff, force_only)
        return force, energy, minimum
    force = depth.pow(1.5)*F.softplus(coeff[:, 0])
    return force, None, None


def train(preprocess, training, dev, device, epochs, decoder=None, noisy=False):
    network = Network(28 if decoder else 1).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0001)
    height, truth, _, valid, _ = training
    mask = torch.as_tensor(valid, device=device)
    validation = dev[0]
    started = time.perf_counter()
    best = (float("inf"), None, 0)
    batch = 8 if decoder else 32
    for epoch in range(epochs):
        for k in torch.randperm(len(truth), device=device).split(batch):
            values = height[k]
            if noisy:
                amplitude = torch.rand((len(k), 1, 1), device=device)*0.025
                values = values+torch.randn_like(values)*amplitude*mask[k]
            force, energy, _ = predict(network, preprocess(values, decoder is not None), decoder)
            error = (force-truth[k])/truth[k].clamp(min=0.20)
            loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.15)
            if energy is not None:
                scale = values.amax((-1, -2)).clamp(min=0.15).square()
                loss = loss+0.001*(energy/scale).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.inference_mode():
            force = torch.cat([predict(network, preprocess(part, decoder is not None), decoder)[0]
                               for part in validation.split(batch)])
            error = ((force-dev[1]).abs()/dev[1].clamp(min=0.20)).mean().item()
        if error<best[0]:
            best = (error, {key: val.detach().clone() for key, val in network.state_dict().items()},
                    epoch+1)
        if (epoch+1) % 10 == 0 or epoch+1 == epochs:
            print(f"{'pinn' if decoder else 'image'} epoch {epoch+1}/{epochs}: "
                  f"dev scaled MAE {error:.5f}, elapsed {time.perf_counter()-started:.1f} s", flush=True)
    sync(device)
    network.load_state_dict(best[1])
    return network, {"development_scaled_mae": best[0], "epoch": best[2],
                     "seconds": time.perf_counter()-started}


def ridge(preprocess, training, development):
    def source(group):
        _, features, depth, _ = preprocess(group[0])
        return features.cpu().numpy(), depth.cpu().numpy()

    with torch.inference_mode():
        x, d = source(training)
        dev, dd = source(development)
    mean = x.mean(0)
    scale = x.std(0).clip(min=0.05)
    x = (x-mean)/scale
    dev = (dev-mean)/scale
    target = np.log(training[1].cpu().numpy()/d.clip(min=0.02)**1.5)
    true = development[1].cpu().numpy()

    best = (float("inf"), None, None)
    for width in (0.5, 1, 2, 4):
        kernel = gaussian(x, x, width)
        for reg in (0.01, 0.05, 0.2):
            coefficient = np.linalg.solve(kernel+reg*np.eye(len(x)), target-target.mean())
            y = dd**1.5*np.exp(gaussian(dev, x, width)@coefficient+target.mean())
            error = np.mean(np.abs(y-true)/true.clip(min=0.2))
            if error<best[0]:
                best = (error, width, coefficient)

    weights = dict(mean=mean, scale=scale, x=x, width=best[1], coefficient=best[2],
                   target_mean=target.mean())
    return weights, float(best[0])


def gaussian(a, b, width):
    return np.exp(-((a[:, None]-b[None])**2).mean(axis=2)/(2*width**2))


def ridge_predict(inputs, parameters):
    _, features, depth, _ = inputs
    data = (features.cpu().numpy()-parameters["mean"])/parameters["scale"]
    kernel = gaussian(data, parameters["x"], parameters["width"])
    gain = (kernel@parameters["coefficient"])[0]+parameters["target_mean"]
    return float(depth.item()**1.5*np.exp(gain))


def evaluate(preprocess, estimator, data, device, physical, noise_um=0, noise_seed=7621):
    heights, forces, travel, mask, labels = data
    if noise_um:
        rng = np.random.default_rng(noise_seed)
        values = rng.normal(0, noise_um/1000, mask.shape).astype(np.float32)*mask
        heights = heights+torch.as_tensor(values, device=device)
    reference = forces.cpu().numpy()
    rows = []
    with torch.inference_mode():
        for _ in range(5):
            estimator(preprocess(heights[:1], physical), physical)
        for i, height in enumerate(heights):
            sync(device)
            start = time.perf_counter()
            inputs = preprocess(height[None], physical)
            force, minimum = estimator(inputs, physical)
            sync(device)
            elapsed = (time.perf_counter()-start)*1000
            difference = 0
            if physical:
                full_force, minimum = estimator(inputs)
                difference = abs(full_force-force)
                assert difference<5e-5, "Top-layer and full-mesh force integrals differ."
            rows.append((labels[i], travel[i], reference[i], float(force), elapsed,
                         float(minimum), difference))
    return rows


def summary(rows):
    labels = [r[0] for r in rows]
    numeric = np.asarray([r[1:] for r in rows])
    travel, true, fit, elapsed, minimum, difference = numeric.T
    err = fit-true
    loaded = true>=0.20
    finals = [k for k in range(len(labels)) if k+1 == len(labels) or labels[k] != labels[k+1]]
    return {
        "trajectories": len(set(labels)), "frames": len(rows), "MAE_N": float(abs(err).mean()),
        "Bias_N": float(err.mean()), "MaxAbs_N": float(abs(err).max()),
        "MAPE_above_0p2N_pct": float(100*np.mean(abs(err[loaded]/true[loaded]))) if loaded.any() else None,
        "MaxFinalAPE_pct": float(100*np.max(abs(err[finals]/true[finals]))),
        "P95_ms": float(np.quantile(elapsed, 0.95)),
        "MinJGauss": float(minimum.min()) if np.isfinite(minimum).all() else None,
        "FastForceDifference_N": float(difference.max()),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_table(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def write_coverage(output, records):
    write_table(output/"coverage.csv", ("Trajectory", "Group", "Frames", "Seconds", "InputSHA256"),
                ([row[key] for key in ("trajectory", "group", "frames", "seconds", "input_sha256")]
                 for row in records))


def fit_models(args, device, protocol):
    groups, records = read_dataset([args.data, *args.extra_data], device, protocol)
    if groups.keys() != HISTORY_GROUPS:
        raise ValueError("Separate training and development trajectories are required.")
    check_disjoint(*(list(row for row in records if row["group"] == group)
                     for group in ("train", "development")))
    implementation = implementation_hashes()
    args.output.mkdir(parents=True, exist_ok=False)
    write_coverage(args.output, records)
    observation = Observations(device)
    details = {}
    checkpoints = {}
    for method in METHODS:
        torch.manual_seed(826)
        if method == "ridge":
            parameters, dev = ridge(observation, groups["train"], groups["development"])
            path = args.output/"ridge.npz"
            np.savez(path, **parameters)
            details[method] = {"development_scaled_mae": dev}
        else:
            decoder = Decoder(device) if method == "pinn" else None
            model, details[method] = train(observation, groups["train"], groups["development"],
                                           device, args.epochs, decoder, args.noise_aug)
            path = args.output/f"{method}.pt"
            torch.save(model.state_dict(), path)
        checkpoints[path.name] = sha256(path)
    if implementation_hashes() != implementation:
        raise ValueError("Implementation changed during training. Start a new run.")
    write_json(args.output/"training.json", details)
    write_json(args.output/"freeze.json", {
        "version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "blind_protocol_sha256": sha256(PROTOCOL), "implementation": implementation,
        "checkpoints": checkpoints, "training_records": records,
        "training_environment": {"device": args.device, "threads": args.threads,
                                 "torch": torch.__version__, "numpy": np.__version__},
        "epochs": args.epochs, "noise_aug": args.noise_aug,
    })
    print(f"Frozen model manifest SHA-256: {sha256(args.output/'freeze.json')}", flush=True)


def score_models(args, device, protocol):
    manifest = args.checkpoint_dir/"freeze.json"
    lock = json.loads(manifest.read_text(encoding="utf-8"))
    if lock["version"] != 1 or lock["implementation"] != implementation_hashes():
        raise ValueError("The test protocol, FEM input mesh or implementation changed after training.")
    if lock["checkpoints"].keys() != {"ridge.npz", "image.pt", "pinn.pt"}:
        raise ValueError("The frozen comparison must include ridge, image and PINN.")
    for name, digest in lock["checkpoints"].items():
        if sha256(args.checkpoint_dir/name) != digest:
            raise ValueError(f"Checkpoint {name} changed after training.")
    token = sha256(manifest)
    groups, records = read_dataset([args.data, *args.extra_data], device, protocol, token)
    check_disjoint(lock["training_records"], records)
    check_disjoint(*(list(row for row in records if row["group"] == group)
                     for group in ("test", "combined")))
    data = groups["combined"]
    keep = data[2] > protocol["deep_threshold_mm"]
    if not keep.any():
        raise ValueError("The prescribed deep subset is empty.")

    args.output.mkdir(parents=True, exist_ok=False)
    write_coverage(args.output, records)
    write_json(args.output/"evaluation.json", {
        "freeze_sha256": token, "blind_protocol": protocol,
        "device": args.device, "threads": args.threads, "torch": torch.__version__,
        "numpy": np.__version__, "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "CPU/GPU single-frame inference including height preprocessing, after five warmups",
    })
    observation = Observations(device)
    report = {}
    for method in METHODS:
        decoder = Decoder(device) if method == "pinn" else None
        if method == "ridge":
            with np.load(args.checkpoint_dir/"ridge.npz") as stored:
                parameters = dict(stored.items())

            def estimator(inputs, fast=False):
                return ridge_predict(inputs, parameters), np.nan
        else:
            model = Network(28 if decoder else 1).to(device).eval()
            weights = torch.load(args.checkpoint_dir/f"{method}.pt", weights_only=True,
                                 map_location=device)
            model.load_state_dict(weights)

            def estimator(inputs, fast=False):
                force, _, j = predict(model, inputs, decoder, fast)
                return force.item(), j.item() if j is not None else np.nan

        report[method] = {}
        for group in ("test", "combined"):
            for noise in (0, protocol["height_noise_um"]):
                label = f"{group}_noise{noise}" if noise else group
                rows = evaluate(observation, estimator, groups[group], device, decoder is not None,
                                noise, protocol["noise_seed"])
                write_table(args.output/f"{method}_{label}.csv", RESULT_COLUMNS, rows)
                report[method][label] = summary(rows)
                if label == "combined":
                    # Reuse exactly these predictions; a subgroup should not trigger a new benchmark.
                    deep = [row for row, selected in zip(rows, keep) if selected]
                    write_table(args.output/f"{method}_combined_beyond_3p12.csv", RESULT_COLUMNS, deep)
                    report[method]["combined_beyond_3p12"] = summary(deep)
                stats = report[method][label]
                print(f"{method:5s} {label:18s} {len(rows):3d} frames  "
                      f"MAE {stats['MAE_N']:.5f} N  P95 {stats['P95_ms']:.3f} ms", flush=True)
    write_json(args.output/"summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "evaluate"):
        command = modes.add_parser(mode)
        command.add_argument("data", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--extra-data", action="append", type=Path, default=[])
        command.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
        command.add_argument("--threads", type=int, default=2)
    modes.choices["train"].add_argument("--epochs", type=int, default=120)
    modes.choices["train"].add_argument("--noise-aug", action="store_true")
    modes.choices["evaluate"].add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or (args.mode == "train" and args.epochs < 1):
        parser.error("Thread and epoch counts must be positive.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.mode == "train":
        fit_models(args, device, protocol)
    else:
        score_models(args, device, protocol)


if __name__ == "__main__":
    main()
