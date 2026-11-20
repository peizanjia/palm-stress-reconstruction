function verify_indentation()
% Verify the 3-D contact tangent, off-axis loading, and the Hertz limit.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'src'),fullfile(root,'config'));
cfg = indentation_config();
cfg.radius = 9;
cfg.thickness = 3;
cfg.mesh.size = 0.85;
cfg.mesh.fineRadius = 2.5;
cfg.probeCenter = [1.2,-0.6];
cfg.penaltyFactor = 280;
cfg.depths = [0.04,0.10,0.16];
model = palm.indent.prepare(cfg);
% For an undeformed plane, the sphere-plane intersection gives a known cap.
planeDepth = 0.30;
[cap,~,penetration] = palm.indent.capDepth(model,cfg,zeros(3*model.n,1),planeDepth);
assert(abs(cap-planeDepth) < 0.004 && abs(penetration-planeDepth) < 0.004);
fprintf('Sphere-plane cap integration error %.3g mm.\n',abs(cap-planeDepth));
u = 0.001*sin((1:3*model.n)');
u(model.fixed) = 0;
depth = 0.20;
v = cos((1:3*model.n)');
v(model.fixed) = 0;
v = v/norm(v);
[~,K] = palm.indent.assemble(model,cfg,u,depth,true);
epsilon = 1e-5;
plus = palm.indent.assemble(model,cfg,u+epsilon*v,depth,false);
minus = palm.indent.assemble(model,cfg,u-epsilon*v,depth,false);
errorRatio = norm((plus-minus)/(2*epsilon)-K*v)/norm(K*v);
assert(errorRatio < 1e-7);
r = run_indentation(cfg);
assert(height(r.history) == numel(cfg.depths));
assert(all(diff(r.history.Force_N)>0) && all(r.history.MinJ>0));
assert(max(abs(r.history.Force_N-r.history.Support_N)) < 1e-6);
[~,peak] = min(r.displacement(r.mesh.topNodes,3));
nearest = r.mesh.X(r.mesh.topNodes(peak),1:2);
assert(norm(nearest-cfg.probeCenter) < cfg.mesh.size);
fprintf('3-D tangent relative error %.3g, eccentric-load force closed.\n',errorRatio);

cfg = indentation_config();
cfg.nu = 0.3;
cfg.radius = 6;
cfg.thickness = 5;
cfg.probeRadius = 1;
cfg.maxCapDepth = 0.2;
cfg.mesh.size = 0.12;
cfg.mesh.fineRadius = 0.58;
cfg.mesh.outerSpacing = 1.5;
cfg.penaltyFactor = 40;
cfg.depths = [0.02,0.04,0.06,0.08];
r = run_indentation(cfg);
E = 2*cfg.mu*(1+cfg.nu);
hertz = 4/3*E/(1-cfg.nu^2)*sqrt(cfg.probeRadius)*cfg.depths(:).^(3/2);
assert(height(r.history) == numel(hertz), r.status);
relative = r.history.Force_N./hertz - 1;
fprintf('Hertz comparison, travel %.3f mm, FEM %.6f N, Hertz %.6f N, error %.2f%%\n', ...
    cfg.depths(end),r.history.Force_N(end),hertz(end),100*relative(end));
assert(max(abs(relative)) < 0.15);
end
