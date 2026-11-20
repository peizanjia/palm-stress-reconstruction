function verify_mixed_indentation()
% Validate the stabilized coupled residual and volume balance in 3-D.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'src'),fullfile(root,'config'));
cfg = indentation_config();
cfg.radius = 9;
cfg.thickness = 3;
cfg.mesh.size = 0.9;
cfg.mesh.fineRadius = 2.5;
cfg.probeCenter = [1.2,-0.6];
cfg.depths = [0.04,0.12,0.2];
cfg.pressureStabilization = .05;
model = palm.indent.prepare(cfg);
S = model.pressureStabilization;
assert(norm(S*ones(model.n,1),inf) < 1e-13);
state = .004*sin((1:model.nState)');
state(model.fixed) = 0;
v = cos((1:model.nState)');
v(model.fixed) = 0;
v = v/norm(v);
depth = 0.2;
[~,K] = palm.indent.assemble(model,cfg,state,depth,true);
epsilon = 1e-5;
plus = palm.indent.assemble(model,cfg,state+epsilon*v,depth,false);
minus = palm.indent.assemble(model,cfg,state-epsilon*v,depth,false);
errorRatio = norm((plus-minus)/(2*epsilon)-K*v) / norm(K*v);
assert(errorRatio < 1e-7);
assert(norm(K-K','fro')/norm(K,'fro') < 1e-12);
r = run_indentation(cfg);
assert(height(r.history)==numel(cfg.depths));
assert(all(r.history.MinJ>0) && all(diff(r.history.Force_N)>0));
assert(max(abs(r.history.Force_N-r.history.Support_N)) < 1e-6);
[meta,trace] = palm.dataset.prepareSurface(model,cfg,0.5);
state = [r.displacement(:);r.multiplier_MPa/model.pressureScale];
h = r.history(end,:);
frame = palm.dataset.snapshot(model,cfg,meta,trace,state,h{1,:});
assert(all(isfinite(frame.cauchy_MPa),'all') && all(isfinite(frame.multiplier_MPa),'all'));
assert(abs(frame.contactForce_N(3)+h.Force_N)<1e-9);
stats = palm.fem.volumeStats(model,r.displacement(:));
assert(abs(stats(1)) < 1e-7);
fprintf('Mixed tangent relative error %.3g, total volume change %.3g mm^3.\n', ...
    errorRatio,stats(1));
end
