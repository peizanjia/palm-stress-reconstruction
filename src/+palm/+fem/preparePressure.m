function fem = preparePressure(fem,mu,gamma)
% Equal-order continuous wedge pressure, stabilized by local mean projection.
% Store pressure coefficients in scaled coordinates: p = pressureScale * w.
fem.pressureScale = mu; % mu divided by a characteristic length of 1 mm
fem.pressureBasis = zeros(6,6);
triangle = [2/3,1/6,1/6; 1/6,2/3,1/6; 1/6,1/6,2/3];
axial = (1+[-1,1]/sqrt(3))/2;
local = zeros(6);
q = 0;
for t = axial
    for l = triangle'
        q = q+1;
        N = [(1-t)*l;t*l];
        fem.pressureBasis(:,q) = N;
        fluctuation = N - 1/6;
        local = local + fluctuation*fluctuation';
    end
end
n = fem.nNodes;
[i,j] = ndgrid(1:6);
rows = fem.wedges(:,i(:))';
cols = fem.wedges(:,j(:))';
entries = reshape(local.*fem.weight,36,[]);
scale = gamma*fem.pressureScale^2/mu;
fem.pressureStabilization = scale*sparse(rows(:),cols(:),entries(:),n,n);
[i,j] = ndgrid(1:18,1:6);
fem.couplingRows = fem.dofs(:,i(:))';
fem.couplingCols = fem.wedges(:,j(:))';
end
