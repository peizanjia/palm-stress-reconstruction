function frame = snapshot(model,cfg,meta,trace,u,diagnostics)
% Save current height and separate the stress trace from the applied traction.
U = reshape(u(1:3*model.n),[],3);
mixed = numel(u)>3*model.n;
if mixed
    nodalPressure = model.pressureScale*u(3*model.n+1:end);
    topPressure = nodalPressure(model.contactNodes);
    frame.multiplier_MPa = topPressure*model.N';
end
X = model.X+U;
frame.currentXYZ_mm = X(meta.nodeIds,:);
triangle = meta.triangles;
a = frame.currentXYZ_mm(triangle(:,2),:) - frame.currentXYZ_mm(triangle(:,1),:);
b = frame.currentXYZ_mm(triangle(:,3),:) - frame.currentXYZ_mm(triangle(:,1),:);
crossAB = cross(a,b,2);
assert(all(crossAB(:,3)>0),'Top surface lost positive orientation in the height projection.');
currentArea = vecnorm(crossAB,2,2)/2;
frame.currentArea_mm2 = currentArea;
frame.outwardNormal = crossAB./(2*currentArea);

surface = triangulation(triangle,frame.currentXYZ_mm(:,1:2));
[id,l] = pointLocation(surface,trace.queryXY);
valid = ~isnan(id);
faceZ = reshape(frame.currentXYZ_mm(triangle(id(valid),:),3),[],3);
z = NaN(size(id));
z(valid) = sum(l(valid,:).*faceZ,2);
frame.height_mm = reshape(z,trace.gridSize);

% Cauchy stress on the top trace of each adjoining volume element.
nFace = size(triangle,1);
nQuad = size(model.N,1);
frame.cauchy_MPa = zeros(nFace,nQuad,6);
frame.stressTraction_MPa = zeros(nFace,nQuad,3);
frame.stressNormal_MPa = zeros(nFace,nQuad);
elementU = permute(reshape(U(trace.wedges',:),6,nFace,3),[1,3,2]);
n = reshape(frame.outwardNormal',3,1,nFace);
for q = 1:nQuad
    F = eye(3)+pagemtimes(elementU,'transpose',trace.grad{q},'none');
    [P,J,Q] = palm.material.neoHookean(F,model.material);
    if mixed
        p = reshape(frame.multiplier_MPa(:,q),1,1,nFace);
        P = P + p.*J.*Q;
    end
    assert(~isempty(P),'Invalid deformation gradient on the top surface.');
    sigma = pagemtimes(P,'none',F,'transpose')./J;
    sigmaComponents = reshape(sigma,9,nFace)';
    frame.cauchy_MPa(:,q,:) = reshape(sigmaComponents(:,[1,5,9,2,3,6]),nFace,1,6);
    traction = reshape(pagemtimes(sigma,n),3,nFace)';
    frame.stressTraction_MPa(:,q,:) = reshape(traction,nFace,1,3);
    frame.stressNormal_MPa(:,q) = sum(traction.*frame.outwardNormal,2);
end

travel = diagnostics(1);
if isfield(cfg,'probeShape')
    [gap,normal,~,points] = palm.indent.shapedGap(model,cfg,X,travel,false);
else
    [gap,normal,~,relative,center] = palm.indent.probeGap(model,cfg,X,travel);
    points = relative+reshape(center,1,1,3);
end
frame.quadratureXYZ_mm = permute(points,[2,1,3]);
frame.gapPenetration_mm = -gap';
pressure = -model.contactStiffness*gap' .* (meta.referenceArea_mm2./currentArea);
frame.contactPressure_MPa = pressure;
if isfield(cfg,'probeShape')
    frame.contactPressure_MPa = pressure.*vecnorm(normal,2,3)';
end
frame.contactTraction_MPa = pressure.*permute(normal,[2,1,3]);
frame.contactActive = gap'<0;
weightedArea = currentArea*meta.quadratureWeights';
frame.contactForce_N = reshape(sum(weightedArea.*frame.contactTraction_MPa,[1,2]),1,3);
frame.travel_mm = travel;
frame.capDepth_mm = diagnostics(2);
frame.force_N = diagnostics(4);
frame.support_N = diagnostics(5);
frame.minJ = diagnostics(7);
frame.residual_N = diagnostics(8);
end
