"""Train a variational PINN and compare an untrained reduced Ritz solve."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
import torch
from torch import nn
from torch.nn import functional as func


ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / "results/stage-01-forward-model"
REFERENCE = FORWARD / "task-03-inverse-reference/20261120_reference_v1"
MESH = FORWARD / "task-04-compressibility/20261120_comparison_v1/compressible045/comparison.mat"
CASES = ("centered", "eccentric")
VALIDATION_STEPS = (3, 7, 11, 15, 18)


class VariationalModel:
    """Neo-Hookean energy on the reference wedge mesh and a compact trial space."""

    def __init__(self, device, smoothing_mm=0):
        record = loadmat(MESH, simplify_cells=True)["r"]
        mesh = record["mesh"]
        self.device = device
        self.smoothing_mm = smoothing_mm
        self.mu = float(record["config"]["mu"])
        nu = float(record["config"]["nu"])
        self.lam = 2 * self.mu * nu / (1 - 2 * nu)
        self.height = float(record["config"]["thickness"])
        self.radius = float(record["config"]["radius"])
        self.nodes = self.tensor(mesh["X"])
        self.wedges = torch.as_tensor(mesh["wedges"] - 1, dtype=torch.long, device=device)
        self.side = torch.as_tensor(mesh["sideNodes"] - 1, dtype=torch.long, device=device)
        self.top = torch.as_tensor(np.setdiff1d(mesh["topNodes"], mesh["sideNodes"]) - 1,
                                   dtype=torch.long, device=device)
        self.fraction = self.nodes[:, 2] / self.height
        self.rim = (1 - self.nodes[:, :2].square().sum(1) / self.radius**2).clamp(min=0)
        self.rim[self.side] = 0
        self.active_node = torch.ones(len(self.nodes), device=device)
        self.active_node[self.side] = 0
        self.eye = torch.eye(3, device=device)
        self.virtual = torch.zeros(len(self.nodes), device=device)
        self.virtual[self.top] = 1
        self.grad, self.weights = self.quadrature()
        self.force_grad = torch.einsum("ea,eqaj->eqj", self.virtual[self.wedges], self.grad)
        virtual_elements = self.virtual[self.wedges].bool().any(dim=1)
        self.virtual_quadrature = (self.wedges[virtual_elements], self.grad[virtual_elements],
                                   self.weights[virtual_elements], self.force_grad[virtual_elements])

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def quadrature(self):
        xyz = self.nodes[self.wedges]
        a, b = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
        area2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        span = xyz[:, 3, 2] - xyz[:, 0, 2]
        dx = torch.stack((a[:, 1] - b[:, 1], b[:, 1], -a[:, 1]), dim=-1)
        dy = torch.stack((b[:, 0] - a[:, 0], -b[:, 0], a[:, 0]), dim=-1)
        grad2 = torch.stack((dx, dy), dim=-1) / area2[:, None, None]
        gradients = []
        for zeta in (0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6):
            for corner in range(3):
                bary = self.tensor([2 / 3 if k == corner else 1 / 6 for k in range(3)])
                axial = bary[None, :, None] / span[:, None, None]
                gradients.append(torch.cat((
                    torch.cat(((1 - zeta) * grad2, -axial), dim=2),
                    torch.cat((zeta * grad2, axial), dim=2),
                ), dim=1))
        return torch.stack(gradients, dim=1), area2 * span / 12

    def fields(self, depths, centers, coefficients):
        relative = self.nodes[None, :, :2] - centers[:, None, :]
        widths = self.tensor([1.5, 3, 6])
        radial = relative[:, :, None, :] / widths[None, None, :, None]
        kernels = torch.exp(-0.5 * radial.square().sum(-1)) * self.rim[None, :, None]
        t = self.fraction[None, :, None]
        scale = depths.amax((-1, -2)).clamp(min=0)
        a = torch.tanh(coefficients) * scale[:, None]
        radial = radial * kernels[..., None]
        top_xy = torch.einsum("bnki,bk->bni", radial, a[:, :3])
        interior_xy = torch.einsum("bnki,bk->bni", radial, a[:, 3:6])
        uxuy = t * (top_xy + (1 - t) * interior_xy)
        xy = (self.nodes[None, :, :2] + top_xy) / self.radius
        vertical = func.grid_sample(depths[:, None], xy[:, None], mode="bicubic",
                                    padding_mode="zeros", align_corners=True)[:, 0, 0]
        correction = torch.einsum("bnk,bk->bn", kernels, a[:, 6:9])
        correction += (2 * t[..., 0] - 1) * torch.einsum("bnk,bk->bn", kernels, a[:, 9:])
        z = -t[..., 0] * vertical * self.active_node + t[..., 0] * (1 - t[..., 0]) * correction
        return torch.cat((uxuy, z[..., None]), dim=-1)

    def evaluate(self, depths, centers, coefficients):
        u = self.fields(depths, centers, coefficients)
        return self.evaluate_displacement(u)

    def evaluate_displacement(self, u, force_only=False):
        wedges, grad, weights, force_grad = (self.virtual_quadrature if force_only else
                                            (self.wedges, self.grad, self.weights, self.force_grad))
        gradient = torch.einsum("beai,eqaj->beqij", u[:, wedges], grad)
        F = gradient + self.eye
        a, b, c = F.unbind(dim=-1)
        cof = torch.stack((torch.linalg.cross(b, c), torch.linalg.cross(c, a),
                           torch.linalg.cross(a, b)), dim=-1)
        J = (a * cof[..., 0]).sum(-1)
        j_floor = 0.1
        safe_j = J.clamp(min=j_floor)
        log_j = safe_j.log()
        P_z = self.mu * F[..., 2, :] + ((self.lam * log_j - self.mu) / safe_j)[..., None] * cof[..., 2, :]
        force = -torch.einsum("beqj,eqj,e->b", P_z, force_grad, weights)
        if force_only:
            return None, force, None
        W = self.mu * (F.square().sum((-1, -2)) - 3) / 2 - self.mu * log_j
        W += self.lam * log_j.square() / 2
        W += 10 * (J - j_floor).clamp(max=0).square()
        energy = torch.einsum("beq,e->b", W, weights)
        return energy, force, J.amin(dim=(1, 2))


def read_case(model, name, canonical=False, noise_um=0, rng=None):
    if isinstance(name, Path):
        sample = loadmat(name, simplify_cells=True)
        heights = np.moveaxis(sample["heights"], -1, 0)
        force = sample["force"]
    else:
        directory = REFERENCE / name
        frames = [loadmat(snapshot, simplify_cells=True)["frame"]
                  for snapshot in sorted((directory / "snapshots").glob("step_*.mat"))]
        heights = np.stack([frame["height_mm"] for frame in frames])
        force = [frame["force_N"] for frame in frames]
    if noise_um:
        heights = heights + rng.normal(0, noise_um/1000, heights.shape)
    raw_depth = model.tensor(np.nan_to_num(model.height - heights, nan=0))
    return (*prepare_observation(model, raw_depth, canonical), model.tensor(force), raw_depth)


def prepare_observation(model, depth, canonical):
    count, size, _ = depth.shape
    axis = torch.linspace(-model.radius, model.radius, size, device=model.device)
    if model.smoothing_mm:
        spacing = 2*model.radius/(size-1)
        half_width = int(np.ceil(3*model.smoothing_mm/spacing))
        offsets = torch.arange(-half_width, half_width+1, device=model.device)*spacing
        kernel = torch.exp(-0.5*(offsets/model.smoothing_mm).square())
        kernel = kernel/kernel.sum()
        filtered = func.conv2d(depth[:, None], kernel[None, None, None, :],
                               padding=(0, half_width))
        depth = func.conv2d(filtered, kernel[None, None, :, None],
                            padding=(half_width, 0))[:, 0]
    index = depth.reshape(count, -1).argmax(1)
    center = torch.stack((axis[index % size], axis[index // size]), dim=1)
    offsets = model.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                            [2, 0], [-2, 0], [0, 2], [0, -2],
                            [3, 0], [-3, 0], [0, 3], [0, -3]])
    xy = (center[:, None, :] + offsets[None, :, :]) / model.radius
    features = func.grid_sample(depth[:, None], xy[:, None], align_corners=True)[:, 0, 0]
    if canonical:
        # Recenter the observed patch on the fine part of the reference mesh.
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        grid = (torch.stack((x, y), dim=-1)[None] + center[:, None, None, :]) / model.radius
        depth = func.grid_sample(depth[:, None], grid, align_corners=True)[:, 0]
        center = torch.zeros_like(center)
    return depth, center, features / model.height


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def network(features, device):
    return nn.Sequential(nn.Linear(features.shape[1], 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 12)).to(device)


def train(model, data, epochs, physics_weight, validation=None):
    depth, center, features, truth, _ = data
    net = network(features, model.device)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    if validation is None:
        train_steps = [i for i in range(1, len(depth)) if i not in VALIDATION_STEPS]
        val = list(VALIDATION_STEPS)
        vd, vc, vf, measured = depth[val], center[val], features[val], truth[val]
    else:
        train_steps = torch.where(truth>0)[0].tolist()
        vd, vc, vf, measured, _ = (item[1:] for item in validation)
    best = (float("inf"), None, None)
    sync(model.device)
    started = time.perf_counter()
    for epoch in range(epochs):
        for group in np.array_split(np.random.permutation(train_steps),
                                    (len(train_steps)+3)//4):
            k = group.tolist()
            energy, force, _ = model.evaluate(depth[k], center[k], net(features[k]))
            relative_force = (force - truth[k]) / truth[k].clamp(min=0.15)
            amplitude = (model.height * features[k, 0]).clamp(min=0.1)
            loss = relative_force.square().mean() + physics_weight * (energy / amplitude.square()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, force, _ = model.evaluate(vd, vc, net(vf))
            error = ((force - measured).abs() / measured.clamp(min=0.15)).mean().item()
        if error < best[0]:
            best = (error, {key: value.detach().clone() for key, value in net.state_dict().items()}, epoch + 1)
    sync(model.device)
    net.load_state_dict(best[1])
    return net, time.perf_counter() - started, best[0], best[2]


def ritz_fit(model, depth, center):
    coefficients = torch.zeros((1, 12), device=model.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficients], max_iter=45, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        energy, _, _ = model.evaluate(depth, center, coefficients)
        energy.sum().backward()
        return energy.sum()

    optimizer.step(closure)
    return coefficients.detach()


def score(model, data, net, output, route, canonical):
    _, _, _, truth, raw_depth = data
    records = []
    for i in range(len(raw_depth)):
        sync(model.device)
        start = time.perf_counter()
        depth, center, features = prepare_observation(model, raw_depth[i:i+1], canonical)
        if net is None:
            coefficients = ritz_fit(model, depth, center)
        else:
            with torch.no_grad():
                coefficients = net(features)
        with torch.no_grad():
            energy, force, minimum_j = model.evaluate(depth, center, coefficients)
        sync(model.device)
        latency = 1000 * (time.perf_counter() - start)
        records.append((i, truth[i].item(), force.item(), minimum_j.item(), energy.item(), latency))
    return write_scores(records, output, route)


def write_scores(records, output, route):
    with (output / f"{route}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("Step", "Reference_N", "Estimated_N", "MinJGauss", "Energy_Nmm", "Latency_ms"))
        writer.writerows(records)
    array = np.asarray(records)[1:]
    error = array[:, 2] - array[:, 1]
    relative = 100 * error / array[:, 1]
    summary = {
        "MAE_N": float(np.mean(abs(error))), "MaxAbs_N": float(np.max(abs(error))),
        "MAPE_pct": float(np.mean(abs(relative))), "DeepError_pct": float(relative[-1]),
        "Median_ms": float(np.median(array[:, 5])),
        "P95_ms": float(np.quantile(array[:, 5], 0.95)),
    }
    if np.isfinite(array[:, 3]).all():
        summary["MinJGauss"] = float(np.min(array[:, 3]))
    return summary


def score_height_interpolation(training, test, output, route):
    train_depth = training[4].amax((-1, -2)).cpu().numpy()
    train_force = training[3].cpu().numpy()
    calibration = PchipInterpolator(train_depth, train_force, extrapolate=False)
    depth = test[4].cpu().numpy()
    truth = test[3].cpu().numpy()
    records = []
    clipped = 0
    for i, frame in enumerate(depth):
        started = time.perf_counter()
        z = frame.max()
        bounded = min(train_depth[-1], max(train_depth[0], z))
        estimated = float(calibration(bounded))
        latency = 1000 * (time.perf_counter() - started)
        clipped += int(bounded != z)
        records.append((i, truth[i], estimated, np.nan, np.nan, latency))
    summary = write_scores(records, output, route)
    summary["ClippedFrames"] = clipped
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--held-out", choices=CASES, default="eccentric")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--canonical-center", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--extra-test-dir", type=Path)
    parser.add_argument("--extra-test", action="append", type=Path, default=[])
    parser.add_argument("--train-trajectory", action="append", type=Path, default=[])
    parser.add_argument("--validation-trajectory", type=Path)
    parser.add_argument("--physics-weight", nargs="+", type=float, default=[0.03])
    parser.add_argument("--neural-only", action="store_true")
    parser.add_argument("--test-noise-um", type=float, default=0)
    parser.add_argument("--noise-seed", type=int, default=179)
    parser.add_argument("--height-filter-mm", type=float, default=0)
    args = parser.parse_args()
    if any(weight<=0 for weight in args.physics_weight):
        parser.error("PINN energy weights must be positive.")
    if not np.isfinite(args.test_noise_um) or args.test_noise_um<0:
        parser.error("Test noise must be finite and nonnegative.")
    if not np.isfinite(args.height_filter_mm) or args.height_filter_mm<0:
        parser.error("Height-filter width must be finite and nonnegative.")
    if args.train_trajectory and not args.validation_trajectory:
        parser.error("Specify a separate validation trajectory for multi-trajectory training.")
    candidate_tests = args.extra_test.copy()
    if args.extra_test_dir:
        candidate_tests += sorted(args.extra_test_dir.glob("*.mat"))
        assert candidate_tests, "No additional sampled trajectories were found."
    paths = args.train_trajectory + candidate_tests
    if args.validation_trajectory:
        paths.append(args.validation_trajectory)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Training, validation and test trajectories must be distinct.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(321)
    np.random.seed(321)
    torch.set_num_threads(2)
    model = VariationalModel(torch.device(args.device), args.height_filter_mm)
    calibration = read_case(model, next(case for case in CASES if case != args.held_out),
                            args.canonical_center)
    train_sources = [calibration] + [read_case(model, path, args.canonical_center)
                                    for path in args.train_trajectory]
    training = tuple(torch.cat(values) for values in zip(*train_sources))
    validation = (read_case(model, args.validation_trajectory, args.canonical_center)
                  if args.validation_trajectory else None)
    noise_rng = np.random.default_rng(args.noise_seed)
    test = read_case(model, args.held_out, args.canonical_center, args.test_noise_um, noise_rng)
    tests = {"": test}
    for path in candidate_tests:
        tests[path.stem+"_"] = read_case(model, path, args.canonical_center,
                                        args.test_noise_um, noise_rng)
    (args.output / "configuration.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    summary = {}
    for route, weights_to_try in (("data_only", [0.0]), ("pinn", args.physics_weight)):
        if args.checkpoint_dir:
            net = network(training[2], model.device)
            weights = torch.load(args.checkpoint_dir / f"{route}.pt", map_location=model.device,
                                 weights_only=True)
            net.load_state_dict(weights)
        else:
            candidates = []
            for weight in weights_to_try:
                torch.manual_seed(321)
                np.random.seed(321)
                trained = train(model, training, args.epochs, weight, validation)
                candidates.append((*trained, weight))
            net, seconds, validation_error, epoch, weight = min(candidates, key=lambda item: item[2])
            torch.save(net.state_dict(), args.output / f"{route}.pt")
            choices = [{"weight": c[4], "train_s": c[1], "validation_scaled_MAE": c[2],
                        "selected_epoch": c[3]} for c in candidates]
            (args.output / f"{route}_selection.json").write_text(
                json.dumps(choices, indent=2), encoding="utf-8")
        for prefix, data in tests.items():
            key = prefix+route
            summary[key] = score(model, data, net, args.output, key, args.canonical_center)
            if not args.checkpoint_dir:
                summary[key].update(train_s=seconds, validation_scaled_MAE=validation_error,
                                    selected_epoch=epoch, physics_weight=weight)
            print(key, summary[key], flush=True)
    if not args.neural_only:
        for prefix, data in tests.items():
            for route, result in (("ritz", score(model, data, None, args.output, prefix+"ritz",
                                                 args.canonical_center)),
                                  ("pchip", score_height_interpolation(calibration, data, args.output,
                                                                     prefix+"pchip"))):
                summary[prefix+route] = result
                print(prefix+route, result, flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
