function [pressure,traction,normal,area] = traction(X,meta,M,nodal)
% Work-conjugate L2 recovery on the reference surface, then push to current area.
f = meta.triangles;
a = X(f(:,2),:)-X(f(:,1),:);
b = X(f(:,3),:)-X(f(:,1),:);
crossAB = cross(a,b,2);
assert(all(crossAB(:,3)>0),'Non-graph surface in contact comparison.');
area = vecnorm(crossAB,2,2)/2;
normal = crossAB./(2*area);
t = M\nodal;
traction = zeros(size(f,1),size(meta.barycentric,1),3);
for k = 1:3
    traction(:,:,k) = reshape(t(f,k),[],3)*meta.barycentric';
end
traction = traction.*(meta.referenceArea_mm2./area);
pressure = -sum(traction.*reshape(normal,[],1,3),3);
end
