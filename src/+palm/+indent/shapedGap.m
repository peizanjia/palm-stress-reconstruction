function [gap,gradient,hessian,points] = shapedGap(model,cfg,x,depth,withTangent)
% Implicit gap, derivative and optional Hessian in laboratory coordinates.
shape = cfg.probeShape;
origin = [cfg.probeCenter,cfg.thickness-depth]-shape.lowest;
n = size(model.contactNodes,1);
k = size(model.N,1);
points = zeros(k,n,3);
for i = 1:3
    points(:,:,i) = model.N*reshape(x(model.contactNodes,i),n,3)';
end
v = (reshape(points,[],3)-origin)*shape.B;
p = shape.power;
radius = max(sum(abs(v).^p,2).^(1/p),1e-12);
m = sign(v).*abs(v).^(p-1);
g = m./radius.^(p-1)*shape.B';
gradient = reshape(shape.scale*g,k,n,3);
gap = reshape(shape.scale*min(radius-1,0),k,n);
hessian = [];
if withTangent
    w = abs(v).^(p-2)./radius.^(p-1);
    hessian = zeros(k,n,3,3);
    for i = 1:3
        for j = i:3
            value = w*(shape.B(i,:).*shape.B(j,:))'-g(:,i).*g(:,j)./radius;
            hessian(:,:,i,j) = reshape(shape.scale*(p-1)*value,k,n);
            hessian(:,:,j,i) = hessian(:,:,i,j);
        end
    end
end
end
