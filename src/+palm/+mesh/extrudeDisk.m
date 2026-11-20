function mesh = extrudeDisk(xy,triangles,side,z)
% Extrude a consistently oriented disk triangulation into straight wedges.
nDisk = size(xy,1);
layers = numel(z)-1;
mesh.X = [repmat(xy, layers + 1, 1), repelem(z, nDisk)];
wedges = [triangles, triangles + nDisk];
mesh.wedges = repmat(wedges, layers, 1) + ...
    repelem((0:layers-1)' * nDisk, size(wedges, 1));
mesh.top = triangles + layers * nDisk;
mesh.bottomNodes = (1:nDisk)';
mesh.sideNodes = reshape(side(:) + (0:layers)*nDisk, [], 1);
mesh.topNodes = (layers*nDisk + (1:nDisk))';
end
