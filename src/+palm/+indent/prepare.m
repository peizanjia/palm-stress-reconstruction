function model = prepare(cfg)
mesh = palm.indent.mesh(cfg);
model = palm.fem.prepare(mesh,cfg.fixSide);
model.mesh = mesh;
model.n = model.nNodes;
model.nState = 3*model.n;
if isfield(cfg,'pressureStabilization')
    model = palm.fem.preparePressure(model,cfg.mu,cfg.pressureStabilization);
    model.free = [model.free; (3*model.n+1:4*model.n)'];
    model.nState = 4*model.n;
end
model.contactNodes = mesh.top;
model.contactDofs = [mesh.top,mesh.top+model.n,mesh.top+2*model.n];

a = 0.059715871789770;
b = 0.470142064105115;
c = 0.797426985353087;
d = 0.101286507323456;
model.N = [1/3,1/3,1/3;a,b,b;b,a,b;b,b,a;c,d,d;d,c,d;d,d,c];
weights = [0.225;repmat(0.132394152788506,3,1);repmat(0.125939180544827,3,1)];
x = mesh.X;
t = mesh.top;
ab = x(t(:,2),:)-x(t(:,1),:);
ac = x(t(:,3),:)-x(t(:,1),:);
area = (ab(:,1).*ac(:,2)-ab(:,2).*ac(:,1))/2;
model.faceWeight = weights;
model.referenceFaceArea = area;
model.contactStiffness = cfg.penaltyFactor*cfg.mu/cfg.mesh.size;
model.contactWeight = model.contactStiffness*(weights*area');
[ii,jj] = ndgrid(1:9);
model.contactRows = model.contactDofs(:,ii(:))';
model.contactCols = model.contactDofs(:,jj(:))';
[ii,jj] = ndgrid(1:3);
model.products = model.N(:,ii(:)).*model.N(:,jj(:));
model.material.mu = cfg.mu;
model.material.lambda = 0;
if model.nState == 3*model.n
    model.material.lambda = 2*cfg.mu*cfg.nu/(1-2*cfg.nu);
end
end
