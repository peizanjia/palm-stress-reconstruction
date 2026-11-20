function result = run_forward(cfg)
%RUN_FORWARD Solve the configured circular silicone disk in reference coordinates.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'), fullfile(root, 'config'));
if nargin == 0
    cfg = forward_config();
end
palm.validateConfig(cfg);
material = cfg.material;
material.lambda = 2 * material.mu * material.nu / (1 - 2 * material.nu);
mesh = palm.mesh.cylinder(cfg.geometry, cfg.mesh);
fem = palm.fem.prepare(mesh, cfg.boundary.fixSide);
[loadVector, loadInfo] = palm.load.topTraction(mesh, cfg.load);
[u, history] = palm.solve.equilibrium(fem, material, loadVector, cfg.solver);
result = palm.output.collect(mesh, fem, material, u, loadVector, history);
result.load = loadInfo;
result.config = cfg;
if cfg.output.write
    directory = string(cfg.output.directory);
    if ~java.io.File(char(directory)).isAbsolute()
        directory = fullfile(root, directory);
    end
    palm.output.writeResult(result, directory);
end
if cfg.solver.verbose
    fprintf('%d nodes, %d wedges, max indentation %.6g mm, min J %.6g\n', ...
        size(mesh.X,1), size(mesh.wedges,1), result.maxIndentation, result.minJ);
    fprintf('Applied Fz %.6g N, support Rz %.6g N, force imbalance %.3g N\n', ...
        result.appliedForce(3), result.supportForce(3), norm(result.forceImbalance));
end
end
