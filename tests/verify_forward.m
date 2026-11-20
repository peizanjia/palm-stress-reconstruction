function verify_forward()
% Verify the reference discretization, its tangent, and the public solver.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root,'src'), fullfile(root,'config'));
cfg = forward_config();
cfg.geometry.radius = 6;
cfg.geometry.thickness = 3;
cfg.mesh.innerRadius = 0;
cfg.mesh.outerSpacing = 1.5;
cfg.mesh.axialLayers = 3;
cfg.solver.verbose = false;
cfg.output.write = false;
cfg.load.radius = 2;
mesh = palm.mesh.cylinder(cfg.geometry, cfg.mesh);
fem = palm.fem.prepare(mesh, true);
material = cfg.material;
material.lambda = 2*material.mu*material.nu/(1-2*material.nu);

% Affine uniaxial strain, constant stress and zero residual at free nodes.
s = 0.92;
U = mesh.X .* [0,0,s-1];
Pzz = material.mu*(s-1/s) + material.lambda*log(s)/s;
uniform = cfg.load;
uniform.profile = "full";
uniform.mode = "peak";
uniform.magnitude = -Pzz;
f = palm.load.topTraction(mesh, uniform);
[internal, K, J] = palm.fem.assemble(fem, material, U(:), true);
assert(abs(J-s) < 1e-13);
assert(abs(palm.fem.minimumJacobian(fem,U(:))-s) < 1e-13);
assert(norm(internal(fem.free)-f(fem.free)) < 1e-12);

% Check the consistent tangent by differentiating the integrated force.
v = sin((1:numel(U))');
v(fem.fixed) = 0;
v = v/norm(v);
epsilon = 1e-5;
plus = palm.fem.assemble(fem, material, U(:)+epsilon*v, false);
minus = palm.fem.assemble(fem, material, U(:)-epsilon*v, false);
derivativeError = norm((plus-minus)/(2*epsilon)-K*v) / norm(K*v);
assert(derivativeError < 1e-7);
assert(norm(K-K','fro') / norm(K,'fro') < 1e-12);

% Fail an element orientation check even when all Gauss points are positive.
U = zeros(size(mesh.X));
U(1,3) = 1.1;
assert(palm.fem.minimumJacobian(fem,U(:)) < 0);
[~,~,gaussJ] = palm.fem.assemble(fem,material,U(:),false);
assert(gaussJ > 0);

cfg.load.magnitude = 0;
zero = run_forward(cfg);
assert(norm(zero.displacement,'fro') == 0 && abs(zero.minJ-1) < 1e-13);

% A smooth central patch, in the linear-response limit.
cfg.load.magnitude = 1e-4;
one = run_forward(cfg);
cfg.load.magnitude = 2e-4;
two = run_forward(cfg);
assert(abs(two.appliedForce(3)+2e-4) < 1e-14);
assert(norm(two.forceImbalance) < 1e-9);
assert(norm(two.reaction(mesh.sideNodes,:), 'fro') > 0);
assert(norm(two.displacement(mesh.sideNodes,:), 'fro') == 0);
assert(abs(two.maxIndentation/(2*one.maxIndentation)-1) < 1e-3);

% Offset uniform patch and bottom-only supports.
cfg.boundary.fixSide = false;
cfg.load.profile = "uniform";
cfg.load.center = [1, 0];
cfg.load.magnitude = 0.1;
offset = run_forward(cfg);
assert(norm(offset.forceImbalance) < 1e-8 && offset.minJ > 0);
assert(abs(offset.appliedForce(3)+0.1) < 1e-14);

% Custom profile with an explicit traction amplitude.
custom = cfg.load;
custom.profile = "custom";
custom.mode = "peak";
custom.custom = @(xy) ones(size(xy,1), 1);
custom.magnitude = uniform.magnitude;
assert(norm(palm.load.topTraction(mesh,custom)-f) < 1e-14);
fprintf('Forward verification passed (relative tangent error %.3g).\n', derivativeError);
end
