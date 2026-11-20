function cfg = indentation_config()
% Three-dimensional Fig. 3c contact model. Units are mm, N, MPa.
cfg.radius = 25;
cfg.thickness = 5;           % Provisional geometry for follow-up studies
cfg.probeRadius = 4;
cfg.probeCenter = [0,0];
cfg.mu = 0.030;
cfg.nu = 0.45;
cfg.fixSide = true;

cfg.mesh.size = 0.53;       % Element size near the contact and upper surface
cfg.mesh.fineRadius = 5;
cfg.mesh.growth = 1.45;
cfg.mesh.outerSpacing = 5;
cfg.penaltyFactor = 140;    % Contact penalty is penaltyFactor * mu / mesh.size
cfg.depths = (0.12:0.20:5.12)'; % Rigid probe travel, not optical cap depth
cfg.maxCapDepth = 2.87;     % Stop after reaching the experimental cap-depth range
cfg.maxIterations = 24;
cfg.minStep = 0.0005;
cfg.forceTolerance = 2e-8;  % N (free DOF residual)
end
