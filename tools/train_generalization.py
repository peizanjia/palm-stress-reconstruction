"""Train force estimators from grouped indentation trajectories."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F

from train_surface_surrogate import VariationalModel, sync


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


def read_dataset(folders, device):
    groups = {}
    records = []
    paths = [path for folder in folders for path in sorted(folder.glob("*.mat"))
             if path.stem != "design"]
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Trajectory identifiers must be unique across data directories.")
    for path in paths:
        mat = loadmat(path, simplify_cells=True, variable_names=[
            "heights", "travel", "force", "spec", "diagnostics", "seconds"])
        spec = mat["spec"]
        file_group = path.stem.split("_")[0]
        group = file_group
        if group in ("train", "development") and 60 <= spec["yaw_deg"] <= 100 and not (
                spec["power"] == 2 and np.ptp(spec["axes"]) < 1e-10):
            group = "yaw"
        travel = np.atleast_1d(mat["travel"])
        force = np.atleast_1d(mat["force"])
        complete = len(travel) == len(np.atleast_1d(spec["depths_mm"]))+1
        records.append((path.stem, group, len(travel)-1, complete, mat["seconds"]))
        if len(travel) <= 1:
            continue
        diagnostics = np.atleast_2d(mat["diagnostics"])
        good = (force > 0) & (diagnostics[:, 6] > 0) & (diagnostics[:, 7] <= 2.1e-8)
        if group == "deep":
            good &= travel>2.7
        array = np.moveaxis(np.atleast_3d(mat["heights"]), -1, 0)[good]
        valid = np.isfinite(array)
        raw = np.where(valid, 5-array, 0)
        groups.setdefault(group, []).append((raw, force[good], travel[good], valid,
                                            np.repeat(path.stem, np.count_nonzero(good))))
    joined = {}
    for group, rows in groups.items():
        z, forces, depth, mask, labels = (np.concatenate(field) for field in zip(*rows))
        if len(forces):
            joined[group] = (torch.as_tensor(z, device=device, dtype=torch.float32),
                             torch.as_tensor(forces, device=device, dtype=torch.float32),
                             depth, mask, labels)
    return joined, records


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


def evaluate(preprocess, estimator, data, device, physical, noise_um=0, fast=False):
    heights, forces, travel, mask, labels = data
    if noise_um:
        rng = np.random.default_rng(7621)
        values = rng.normal(0, noise_um/1000, mask.shape).astype(np.float32)*mask
        heights = heights+torch.as_tensor(values, device=device)
    reference = forces.cpu().numpy()
    rows = []
    with torch.inference_mode():
        estimator(preprocess(heights[:1], physical), fast)
        for i, height in enumerate(heights):
            sync(device)
            start = time.perf_counter()
            inputs = preprocess(height[None], physical)
            force, minimum = estimator(inputs, fast)
            sync(device)
            elapsed = (time.perf_counter()-start)*1000
            difference = 0
            if fast and physical:
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


def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("data", type=Path)
    cmd.add_argument("output", type=Path)
    cmd.add_argument("--epochs", type=int, default=120)
    cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cmd.add_argument("--paths", nargs="+", choices=("ridge", "image", "pinn"), default=["ridge","image","pinn"])
    cmd.add_argument("--noise-aug", action="store_true")
    cmd.add_argument("--checkpoint-dir", type=Path)
    cmd.add_argument("--extra-data", action="append", type=Path, default=[])
    cmd.add_argument("--fast-force", action="store_true")
    args = cmd.parse_args()
    if args.epochs<1:
        cmd.error("Epoch count must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"configuration.json").write_text(json.dumps(vars(args), indent=2, default=str),
                                                   encoding="utf-8")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    groups, records = read_dataset([args.data, *args.extra_data], device)
    with (args.output/"coverage.csv").open("w", newline="") as file:
        w = csv.writer(file)
        w.writerow(("Trajectory","Group","ConvergedFrames","Complete","Seconds"))
        w.writerows(records)
    observation = Observations(device)
    for key in ("deep", "combined"):
        if key in groups and np.any(groups[key][2]>3.12):
            dataset = groups[key]
            keep = dataset[2]>3.12
            tensor_mask = torch.as_tensor(keep, device=device)
            groups[key+"_beyond_3p12"] = (dataset[0][tensor_mask], dataset[1][tensor_mask],
                                         *(field[keep] for field in dataset[2:]))
    report = {}
    for method in args.paths:
        torch.manual_seed(826)
        started = time.perf_counter()
        decoder = Decoder(device) if method=="pinn" else None
        if method=="ridge":
            if args.checkpoint_dir:
                with np.load(args.checkpoint_dir/"ridge.npz") as stored:
                    parameters = dict(stored.items())
                details = {}
            else:
                parameters, dev = ridge(observation, groups["train"], groups["development"])
                np.savez(args.output/"ridge.npz", **parameters)
                details = {"development_scaled_mae": dev, "seconds": time.perf_counter()-started}

            def estimator(inputs, fast=False):
                return ridge_predict(inputs, parameters), np.nan
        else:
            if args.checkpoint_dir:
                model = Network(28 if decoder else 1).to(device)
                weights = torch.load(args.checkpoint_dir/f"{method}.pt", weights_only=True,
                                     map_location=device)
                model.load_state_dict(weights)
                details = {}
            else:
                model, details = train(observation, groups["train"], groups["development"], device,
                                       args.epochs, decoder, args.noise_aug)
                torch.save(model.state_dict(), args.output/f"{method}.pt")

            def estimator(inputs, fast=False):
                force, _, j = predict(model, inputs, decoder, fast)
                return force.item(), j.item() if j is not None else np.nan

        report[method] = {"training": details}
        evaluations = [(key,key,0) for key in groups if key!="train"]
        evaluations += [(key+"_noise25",key,25) for key in ("test", "combined") if key in groups]
        for label, group, noise in evaluations:
            rows = evaluate(observation, estimator, groups[group], device, decoder is not None, noise,
                            args.fast_force)
            with (args.output/f"{method}_{label}.csv").open("w", newline="") as file:
                w = csv.writer(file)
                w.writerow(("Trajectory","Travel_mm","Reference_N","Estimate_N","Latency_ms",
                            "MinJGauss","FastForceDifference_N"))
                w.writerows(rows)
            report[method][label] = summary(rows)
            print(method, label, report[method][label], flush=True)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
