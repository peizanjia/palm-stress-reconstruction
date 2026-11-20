function mesh = cylinder(geometry, options)
% Triangulate a convex polygon inscribed in the disk, then extrude prisms.
R = geometry.radius;
inner = linspace(0, options.innerRadius, ...
    max(1, ceil(options.innerRadius / options.innerSpacing)) + 1);
outer = linspace(options.innerRadius, R, ...
    max(1, ceil((R - options.innerRadius) / options.outerSpacing)) + 1);
radii = unique([inner, outer]);
xy = [0, 0];
for ring = 2:numel(radii)
    r = radii(ring);
    spacing = r - radii(ring - 1);
    n = max(6, ceil(2*pi*r / spacing));
    if ring == numel(radii)
        n = max(24, n);
    end
    angles = (0:n-1)' * (2*pi/n) + mod(ring, 2) * pi/n;
    first = size(xy, 1) + 1;
    xy = [xy; r * [cos(angles), sin(angles)]]; %#ok<AGROW>
end
nDisk = size(xy, 1);
side = first:nDisk;
triangles = delaunay(xy(:,1), xy(:,2));
ab = xy(triangles(:,2),:) - xy(triangles(:,1),:);
ac = xy(triangles(:,3),:) - xy(triangles(:,1),:);
reverse = ab(:,1).*ac(:,2) - ab(:,2).*ac(:,1) < 0;
triangles(reverse,[2,3]) = triangles(reverse,[3,2]);

z = linspace(0,geometry.thickness,options.axialLayers+1)';
mesh = palm.mesh.extrudeDisk(xy,triangles,side,z);
end
