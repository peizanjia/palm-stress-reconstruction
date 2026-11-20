function [meta,trace] = prepareSurface(model,cfg,gridSpacing)
% Surface topology and gradients on the silicone side of the top boundary.
nodes = model.mesh.topNodes;
faces = model.mesh.top;
n = size(faces,1);
wedges = model.wedges(end-n+1:end,:);
assert(isequal(wedges(:,4:6),faces),'Top triangles must match the last wedge layer.');
trace.wedges = wedges;

[found,meta.triangles] = ismember(faces,nodes);
assert(all(found,'all'));
meta.nodeIds = nodes;
meta.referenceXYZ_mm = model.X(nodes,:);
meta.barycentric = model.N;
meta.quadratureWeights = model.faceWeight;
meta.referenceArea_mm2 = model.referenceFaceArea;
meta.stressComponents = ["xx","yy","zz","xy","xz","yz"];
meta.config = cfg;
meta.x_mm = -cfg.radius:gridSpacing:cfg.radius;
meta.y_mm = meta.x_mm;
[x,y] = meshgrid(meta.x_mm,meta.y_mm);
trace.queryXY = [x(:),y(:)];
trace.gridSize = size(x);

a = model.X(faces(:,2),:) - model.X(faces(:,1),:);
b = model.X(faces(:,3),:) - model.X(faces(:,1),:);
twiceArea = 2*model.referenceFaceArea;
dX = [a(:,2)-b(:,2),b(:,2),-a(:,2)]./twiceArea;
dY = [b(:,1)-a(:,1),-b(:,1),a(:,1)]./twiceArea;
invHeight = 1./(model.X(wedges(:,4),3)-model.X(wedges(:,1),3));
grad = zeros(6,3,n);
grad(4:6,1,:) = reshape(dX',3,1,n);
grad(4:6,2,:) = reshape(dY',3,1,n);
trace.grad = cell(size(model.N,1),1);
for q = 1:size(model.N,1)
    dZ = model.N(q,:).*invHeight;
    grad(:,3,:) = reshape([-dZ,dZ]',6,1,n);
    trace.grad{q} = grad;
end
end
