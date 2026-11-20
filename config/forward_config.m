function cfg = forward_config()
% Consistent units: mm, N, MPa (N/mm^2).
cfg.geometry.radius = 25;        % Reported palm radius, used for the silicone disk
cfg.geometry.thickness = 5;      % Assumed demo value; see docs/paper-evidence.md
cfg.material.mu = 0.030;         % Illustrative shear modulus, not a Wesitru fit
cfg.material.nu = 0.45;          % Assumed small-strain Poisson ratio
cfg.boundary.fixSide = true;     % Bottom is always fixed

% Delaunay disk sampling and uniform axial extrusion.
cfg.mesh.innerRadius = 6;
cfg.mesh.innerSpacing = 0.9;
cfg.mesh.outerSpacing = 2.8;
cfg.mesh.axialLayers = 6;

% Vertically downward dead traction per unit reference area.
% Modes: "force" [N] rescales the integrated field, "peak" [MPa]
% scales the supplied unit-peak profile directly.
cfg.load.mode = "force";
cfg.load.magnitude = 0.5;
cfg.load.profile = "smooth";     % "smooth", "uniform", "full", or "custom"
cfg.load.center = [0, 0];
cfg.load.radius = 3;            % Prescribed patch radius, independent of the probe radius
cfg.load.custom = [];           % For "custom": @(xy) returns N-by-1 weights in [0, 1]

cfg.solver.initialSteps = 5;
cfg.solver.minStep = 1 / 1024;
cfg.solver.maxIterations = 18;
cfg.solver.maxBacktracks = 12;
cfg.solver.relativeTolerance = 1e-8;
cfg.solver.absoluteTolerance = 1e-11;  % N, norm of residual on free DOFs
cfg.solver.verbose = true;
cfg.output.directory = "results/forward";
cfg.output.write = true;
end
