function summary = verify_inverse_reference(directory)
% Verify an archived dataset by reading every snapshot back from disk.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'config'));
study = inverse_reference_config();
metrics = zeros(numel(study.caseNames),11);
commonAxes = [];
for k = 1:numel(study.caseNames)
    path = fullfile(directory,study.caseNames(k));
    manifest = readtable(fullfile(path,'manifest.csv'),'TextType','string');
    data = load(fullfile(path,'surface_mesh.mat'),'meta');
    meta = data.meta;
    cfg = meta.config;
    assert(isequal(cfg.probeCenter,study.probeCenters(k,:)));
    assert(isequal(manifest.Travel_mm,[0;study.indent.depths(:)]) || ...
        norm(manifest.Travel_mm-[0;study.indent.depths(:)],inf)<1e-12);
    assert(height(manifest) == numel(dir(fullfile(path,'snapshots','*.mat'))));
    assert(all(diff(manifest.Force_N)>0) && all(manifest.MinJ>0));
    assert(all(manifest.Residual_N < cfg.forceTolerance));
    assert(isequal(manifest.Step,(0:height(manifest)-1)'));
    if k == 1
        commonAxes = [meta.x_mm;meta.y_mm];
    else
        assert(isequal(commonAxes,[meta.x_mm;meta.y_mm]));
    end
    assert(max(abs(manifest.Force_N-manifest.Support_N)) < 1e-6);
    [x,y] = meshgrid(meta.x_mm,meta.y_mm);
    for j = 1:height(manifest)
        entry = load(fullfile(path,manifest.Snapshot(j)),'frame');
        frame = entry.frame;
        assert(abs(frame.travel_mm-manifest.Travel_mm(j)) < 1e-12);
        assert(abs(frame.capDepth_mm-manifest.CapDepth_mm(j)) < 1e-12);
        assert(abs(frame.force_N-manifest.Force_N(j)) < 1e-10);
        assert(isequal(size(frame.height_mm),size(x)));
        assert(any(isnan(frame.height_mm),'all') && any(isfinite(frame.height_mm),'all'));
        tri = triangulation(meta.triangles,frame.currentXYZ_mm(:,1:2));
        query = [x(:),y(:)];
        [face,bary] = pointLocation(tri,query);
        active = ~isnan(face);
        assert(isequal(active,isfinite(frame.height_mm(:))));
        vertices = reshape(frame.currentXYZ_mm(meta.triangles(face(active),:),3),[],3);
        assert(norm(frame.height_mm(active)-sum(bary(active,:).*vertices,2),inf) < 1e-12);
        assert(all(isfinite(frame.cauchy_MPa),'all'));
        assert(isequal(size(frame.cauchy_MPa),[size(meta.triangles,1),size(meta.barycentric,1),6]));
        assert(all(frame.contactPressure_MPa>=0,'all'));
        assert(isequal(frame.contactActive,frame.contactPressure_MPa>0));
        weights = frame.currentArea_mm2*meta.quadratureWeights';
        force = reshape(sum(weights.*frame.contactTraction_MPa,[1,2]),1,3);
        assert(norm(force-frame.contactForce_N,inf)<1e-12);
        assert(abs(force(3)+frame.force_N)<1e-9);
        if j == 1
            heightError = max(abs(frame.height_mm(active)-cfg.thickness));
            assert(heightError < 1e-12 && max(abs(frame.cauchy_MPa),[],'all') < 1e-12);
            assert(~any(frame.contactActive,'all'));
            assert(isequal(frame.currentXYZ_mm,meta.referenceXYZ_mm));
        elseif j == 2
            verticalForce = -weights.*frame.contactTraction_MPa(:,:,3);
            centroid = reshape(sum(verticalForce.*frame.quadratureXYZ_mm(:,:,1:2),[1,2]),1,2) ...
                / sum(verticalForce,'all');
            assert(norm(centroid-cfg.probeCenter) < cfg.mesh.size);
        end
    end
    imbalance = max(abs(manifest.Force_N-manifest.Support_N));
    maxDown = cfg.thickness-min(frame.height_mm,[],'all','omitnan');
    metrics(k,:) = [cfg.probeCenter,height(manifest),manifest.Travel_mm(end), ...
        manifest.CapDepth_mm(end),manifest.Force_N(end),maxDown,min(manifest.MinJ), ...
        max(manifest.Residual_N),imbalance,max(manifest.Penetration_mm)];
    fprintf('%s verified %d snapshots, final %.6g N, force imbalance %.3g N.\n', ...
        study.caseNames(k),height(manifest),manifest.Force_N(end), ...
        imbalance);
end

summary = array2table(metrics,'VariableNames',{'ProbeX_mm','ProbeY_mm','Snapshots', ...
    'FinalTravel_mm','FinalCapDepth_mm','FinalForce_N','FinalMaxDown_mm','MinJ', ...
    'MaxResidual_N','MaxForceImbalance_N','MaxPenetration_mm'});
summary = addvars(summary,study.caseNames','Before',1,'NewVariableNames','Case');

% Manufactured affine deformation checks the silicone-side boundary trace.
cfg = study.indent;
cfg.radius = 4;
cfg.mesh.fineRadius = 2;
cfg.mesh.size = 1;
model = palm.indent.prepare(cfg);
[meta,trace] = palm.dataset.prepareSurface(model,cfg,0.5);
F = [1.02,0.01,0.06;0,0.98,0;0,0,0.95];
U = model.X*(F-eye(3))';
[P,J] = palm.material.neoHookean(F,model.material);
sigma = P*F'/J;
expected = reshape(sigma([1,5,9,2,3,6]),1,1,[]);
frame = palm.dataset.snapshot(model,cfg,meta,trace,U(:),[0,0,0,0,0,0,J,0]);
assert(max(abs(frame.cauchy_MPa-expected),[],'all') < 1e-12);
fprintf('Affine upper-surface Cauchy trace verified.\n');
end
