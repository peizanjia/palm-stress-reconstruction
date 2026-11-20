function validateConfig(cfg)
positiveScalar = {'real', 'finite', 'scalar', 'positive'};
positiveInteger = [positiveScalar, {'integer'}];
validateattributes(cfg.geometry.radius, {'numeric'}, positiveScalar);
validateattributes(cfg.geometry.thickness, {'numeric'}, positiveScalar);
validateattributes(cfg.material.mu, {'numeric'}, positiveScalar);
validateattributes(cfg.material.nu, {'numeric'}, ...
    {'real', 'finite', 'scalar', '>=', 0, '<', 0.5});
if cfg.material.nu >= 0.48
    warning('palm:VolumetricLocking', ...
        'Linear displacement wedges may lock near incompressibility. Verify mesh sensitivity.');
end
validateattributes(cfg.boundary.fixSide, {'logical'}, {'scalar'});
validateattributes(cfg.mesh.innerRadius, {'numeric'}, ...
    {'real', 'finite', 'scalar', '>=', 0, '<=', cfg.geometry.radius});
validateattributes(cfg.mesh.innerSpacing, {'numeric'}, positiveScalar);
validateattributes(cfg.mesh.outerSpacing, {'numeric'}, positiveScalar);
validateattributes(cfg.mesh.axialLayers, {'numeric'}, positiveInteger);
mode = string(cfg.load.mode);
profile = string(cfg.load.profile);
if ~isscalar(mode) || ~ismember(mode, ["force", "peak"])
    error('palm:LoadMode', 'Load mode must be "force" or "peak".');
end
if ~isscalar(profile) || ~ismember(profile, ["smooth", "uniform", "full", "custom"])
    error('palm:LoadProfile', 'Load profile must be "smooth", "uniform", "full", or "custom".');
end
validateattributes(cfg.load.magnitude, {'numeric'}, {'real', 'finite', 'scalar', 'nonnegative'});
if ismember(profile, ["smooth", "uniform"])
    validateattributes(cfg.load.center, {'numeric'}, {'real', 'finite', 'size', [1, 2]});
    validateattributes(cfg.load.radius, {'numeric'}, positiveScalar);
    if norm(cfg.load.center) + cfg.load.radius > cfg.geometry.radius
        error('palm:OutsideDisk', 'The prescribed circular patch must fit inside the disk.');
    end
end
if profile == "custom" && ~isa(cfg.load.custom, 'function_handle')
    error('palm:CustomLoad', 'The custom profile requires a function handle.');
end
validateattributes(cfg.solver.initialSteps, {'numeric'}, positiveInteger);
validateattributes(cfg.solver.minStep, {'numeric'}, ...
    [positiveScalar, {'<=', 1/cfg.solver.initialSteps}]);
validateattributes(cfg.solver.maxIterations, {'numeric'}, positiveInteger);
validateattributes(cfg.solver.maxBacktracks, {'numeric'}, positiveInteger);
validateattributes(cfg.solver.relativeTolerance, {'numeric'}, [positiveScalar, {'<', 1}]);
validateattributes(cfg.solver.absoluteTolerance, {'numeric'}, positiveScalar);
validateattributes(cfg.solver.verbose, {'logical'}, {'scalar'});
validateattributes(cfg.output.write, {'logical'}, {'scalar'});
end
