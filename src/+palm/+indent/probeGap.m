function [gap,normal,rho,q,center] = probeGap(model,cfg,x,depth)
% Sphere-relative positions are ordered [quadrature point, triangle, xyz].
center = [cfg.probeCenter,cfg.thickness+cfg.probeRadius-depth];
nTriangles = size(model.contactNodes,1);
q = zeros(size(model.N,1),nTriangles,3);
for i = 1:3
    q(:,:,i) = model.N*reshape(x(model.contactNodes,i),nTriangles,3)'-center(i);
end
rho = max(sqrt(sum(q.^2,3)),eps);
gap = min(rho-cfg.probeRadius,0);
normal = q./rho;
end
