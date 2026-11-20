function [f, info] = topTraction(mesh, options)
% Integrate the vertical nominal traction on the undeformed top triangles.
n = size(mesh.X,1);
tri = mesh.top;
xy = mesh.X(:,1:2);
ab = xy(tri(:,2),:) - xy(tri(:,1),:);
ac = xy(tri(:,3),:) - xy(tri(:,1),:);
area = (ab(:,1).*ac(:,2) - ab(:,2).*ac(:,1)) / 2;
% Seven-point, degree-five triangle quadrature (weights sum to one).
a = 0.059715871789770;
b = 0.470142064105115;
c = 0.797426985353087;
d = 0.101286507323456;
points = [1/3,1/3,1/3; a,b,b; b,a,b; b,b,a; c,d,d; d,c,d; d,d,c];
weights = [0.225, repmat(0.132394152788506,1,3), repmat(0.125939180544827,1,3)];
nodal = zeros(size(tri));
for q = 1:numel(weights)
    l = points(q,:);
    location = l(1)*xy(tri(:,1),:) + l(2)*xy(tri(:,2),:) + l(3)*xy(tri(:,3),:);
    shape = loadProfile(location, options);
    nodal = nodal + (area.*shape*weights(q)) * l;
end
effectiveArea = sum(nodal, 'all');
scale = options.magnitude;
if scale > 0 && effectiveArea <= 0
    error('palm:UnresolvedLoad', 'Load patch was not sampled. Refine the disk near the patch.');
end
if strcmp(options.mode, "force") && scale > 0
    scale = scale / effectiveArea;
end
fz = -scale * accumarray(tri(:), nodal(:), [n,1]);
f = [zeros(2*n,1); fz];
info.effectiveArea = effectiveArea;
info.tractionScale = scale;
end

function shape = loadProfile(xy, options)
switch string(options.profile)
    case "full"
        shape = ones(size(xy,1), 1);
    case "custom"
        shape = options.custom(xy);
        validateattributes(shape, {'numeric'}, ...
            {'size', [size(xy,1),1], 'real', 'finite', '>=', 0, '<=', 1});
    otherwise
        radiusSquared = sum((xy - options.center).^2, 2) / options.radius^2;
        if strcmp(options.profile, "uniform")
            shape = double(radiusSquared <= 1);
        else
            shape = max(1 - radiusSquared, 0).^2;
        end
end
end
