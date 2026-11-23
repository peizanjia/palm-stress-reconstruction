function cfg = generalizationConfig(spec)
% Reconstruct the solver configuration for an archived trajectory.
cfg = indentation_config();
cfg.probeCenter = spec.center_mm;
cfg.probeShape = palm.indent.superellipsoid(spec.axes,spec.power,spec.yaw_deg,spec.tilt_deg);
cfg.depths = spec.depths_mm;
cfg.mesh.size = spec.mesh_mm;
cfg.mesh.fineRadius = 5.5;
cfg.maxIterations = 30;
end
