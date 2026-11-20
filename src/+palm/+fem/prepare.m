function fem = prepare(mesh, fixSide)
% Cache wedge shape gradients and weights in the reference configuration.
wedges = mesh.wedges;
n = size(mesh.X, 1);
fem.nNodes = n;
fem.X = mesh.X;
fem.wedges = wedges;
fem.dofs = [wedges, wedges + n, wedges + 2*n];
fixed = mesh.bottomNodes;
if fixSide
    fixed = union(fixed, mesh.sideNodes);
end
fem.fixed = reshape(fixed + n*(0:2), [], 1);
fem.free = setdiff((1:3*n)', fem.fixed);
a = mesh.X(wedges(:,2),:) - mesh.X(wedges(:,1),:);
b = mesh.X(wedges(:,3),:) - mesh.X(wedges(:,1),:);
height = mesh.X(wedges(:,4),3) - mesh.X(wedges(:,1),3);
twiceArea = a(:,1).*b(:,2) - a(:,2).*b(:,1);
assert(all(twiceArea > 0 & height > 0), 'Invalid reference wedge.');
invHeight = reshape(1 ./ height, 1, 1, []);
gradientXY = permute(cat(3, [a(:,2)-b(:,2), b(:,2), -a(:,2)] ./ twiceArea, ...
    [b(:,1)-a(:,1), -b(:,1), a(:,1)] ./ twiceArea), [2,3,1]);
trianglePoints = [2/3,1/6,1/6; 1/6,2/3,1/6; 1/6,1/6,2/3];
axialPoints = (1 + [-1,1]/sqrt(3)) / 2;
fem.grad = cell(6,1);
q = 0;
for t = axialPoints
    for l = trianglePoints'
        q = q + 1;
        fem.grad{q} = shapeGradient(gradientXY, invHeight, l, t);
    end
end
fem.centroidGrad = shapeGradient(gradientXY, invHeight, ones(3,1)/3, 0.5);
fem.referenceDet = twiceArea .* height;
fem.weight = reshape(fem.referenceDet / 12, 1, 1, []);
[i,j] = ndgrid(1:18);
fem.rows = fem.dofs(:,i(:))';
fem.cols = fem.dofs(:,j(:))';
end

function g = shapeGradient(gradientXY, invHeight, l, t)
dZ = l .* invHeight;
g = [[(1-t)*gradientXY; t*gradientXY], [-dZ; dZ]];
end
