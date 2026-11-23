function verify_shaped_indentation()
% Check equivalence to the spherical law and the tilted shape tangent.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'config'),fullfile(root,'src'));
cfg = indentation_config();
cfg.mesh.size = 0.95;
model = palm.indent.prepare(cfg);
state = zeros(model.nState,1);
depth = 0.63;
[ref,Kref] = palm.indent.assemble(model,cfg,state,depth,true);
cfg.probeShape = palm.indent.superellipsoid([4,4,4],2,37,23);
[sphere,Ksphere] = palm.indent.assemble(model,cfg,state,depth,true);
assert(norm(sphere-ref)<1e-10 && norm(Ksphere-Kref,'fro')<1e-9);

cfg.probeShape = palm.indent.superellipsoid([4.3,2.9,3.4],4,37,23);
[~,K] = palm.indent.assemble(model,cfg,state,depth,true);
direction = sin((1:numel(state))');
direction = direction/norm(direction);
step = 1e-5;
plus = palm.indent.assemble(model,cfg,state+step*direction,depth,false);
minus = palm.indent.assemble(model,cfg,state-step*direction,depth,false);
error = norm((plus-minus)/(2*step)-K*direction)/norm(K*direction);
assert(error<1e-6,'The contact tangent did not match finite differences.');
fprintf('Shaped probe verification passed. Relative tangent error %.3g.\n',error);
end
