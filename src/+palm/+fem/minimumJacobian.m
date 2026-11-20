function minimum = minimumJacobian(fem, u)
% For these straight reference wedges J is linear over the triangle and
% quadratic along each of its three edges parallel to the reference axis.
x = fem.X + reshape(u, [], 3);
nodes = fem.wedges;
e1 = x(nodes(:,2),:) - x(nodes(:,1),:);
e2 = x(nodes(:,3),:) - x(nodes(:,1),:);
d1 = x(nodes(:,5),:) - x(nodes(:,4),:) - e1;
d2 = x(nodes(:,6),:) - x(nodes(:,4),:) - e2;
constant = cross(e1, e2, 2);
linear = cross(d1, e2, 2) + cross(e1, d2, 2);
quadratic = cross(d1, d2, 2);
minimumByElement = inf(size(nodes,1),1);
for j = 1:3
    dz = x(nodes(:,j+3),:) - x(nodes(:,j),:);
    a = dot(quadratic, dz, 2);
    b = dot(linear, dz, 2);
    c = dot(constant, dz, 2);
    edgeMinimum = min(c, a+b+c);
    interior = a > 0 & b < 0 & -b < 2*a;
    edgeMinimum(interior) = c(interior) - b(interior).^2 ./ (4*a(interior));
    minimumByElement = min(minimumByElement, edgeMinimum ./ fem.referenceDet);
end
minimum = min(minimumByElement);
if any(~isfinite(x), 'all')
    minimum = -inf;
end
end
