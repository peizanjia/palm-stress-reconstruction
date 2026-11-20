function mesh = mesh(cfg)
% Locally refined Delaunay footprint and graded axial wedge layers.
h = cfg.mesh.size;
core = cfg.mesh.fineRadius;
center = cfg.probeCenter;
R = cfg.radius;
radial = linspace(0,core,ceil(core/h)+1);
radial = [radial, core+gradedEdges(R+norm(center)-core,h,cfg.mesh.growth,cfg.mesh.outerSpacing)];
xy = center;
for j = 2:numel(radial)
    r = radial(j);
    n = max(6,ceil(2*pi*r/(radial(j)-radial(j-1))));
    theta = (0:n-1)'*2*pi/n + mod(j,2)*pi/n;
    points = center + r*[cos(theta),sin(theta)];
    xy = [xy;points(vecnorm(points,2,2) < R-0.3*h,:)]; %#ok<AGROW>
end
n = max(32,ceil(2*pi*R/cfg.mesh.outerSpacing));
theta = (0:n-1)'*2*pi/n;
firstSide = size(xy,1)+1;
xy = [xy; R*[cos(theta),sin(theta)]];
tri = delaunay(xy(:,1),xy(:,2));
ab = xy(tri(:,2),:)-xy(tri(:,1),:);
ac = xy(tri(:,3),:)-xy(tri(:,1),:);
reverse = ab(:,1).*ac(:,2)-ab(:,2).*ac(:,1) < 0;
tri(reverse,[2,3]) = tri(reverse,[3,2]);
down = [0,gradedEdges(cfg.thickness,h,cfg.mesh.growth,2*cfg.mesh.outerSpacing)];
z = cfg.thickness - flip(down(:));
z(1) = 0;
mesh = palm.mesh.extrudeDisk(xy,tri,firstSide:size(xy,1),z);
end

function edges = gradedEdges(length,h,growth,limit)
edges = [];
position = 0;
while position < length
    position = position+h;
    edges(end+1) = position; %#ok<AGROW>
    h = min(limit,h*growth);
end
edges = edges*(length/edges(end));
edges(end) = length;
end
