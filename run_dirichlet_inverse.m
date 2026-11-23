function directory = run_dirichlet_inverse(runId,caseName,mode,smoothing_mm)
% Stage 02/task 01: recover top reactions from Dirichlet displacement data.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
caseName = validatestring(caseName,{'centered','eccentric'});
mode = validatestring(mode,{'full','height','surface'});
assert(~isempty(regexp(char(runId),'^[A-Za-z0-9][A-Za-z0-9_-]*$','once')));
source = fullfile(root,'results','stage-01-forward-model','task-03-inverse-reference', ...
    '20261120_reference_v1',caseName);
directory = fullfile(root,'results','stage-02-inverse','task-01-dirichlet-reconstruction', ...
    runId,caseName,mode);
assert(~isfolder(directory),'Use a new run identifier.');
mkdir(directory);
data = load(fullfile(source,'surface_mesh.mat'),'meta');
meta = data.meta;
manifest = readtable(fullfile(source,'manifest.csv'),'TextType','string');
cfg = meta.config;
model = palm.indent.prepare(cfg);
nodes = meta.nodeIds;
faces = meta.triangles;
assert(isequal(model.mesh.topNodes,nodes) && isequal(model.X(nodes,:),meta.referenceXYZ_mm));
assert(isequal(model.mesh.top,reshape(nodes(faces),size(faces))));
interiorTop = ~ismember(nodes,model.mesh.sideNodes);
topDofs = nodes(interiorTop)+model.n*(0:2);
[i,j] = ndgrid(1:3);
M = sparse(reshape(faces(:,i(:)),[],1),reshape(faces(:,j(:)),[],1), ...
    reshape(meta.referenceArea_mm2.*(1+(i(:)==j(:))')/12,[],1),numel(nodes),numel(nodes));
u = zeros(3*model.n,1);
[Xg,Yg] = meshgrid(meta.x_mm,meta.y_mm);
grid = [Xg(:),Yg(:)];
spacing = meta.x_mm(2)-meta.x_mm(1);
rows = zeros(height(manifest),24);
series = cell(height(manifest),1);
surfaceMode = strcmp(mode,'surface');
if surfaceMode
    if nargin<4
        smoothing_mm = 2*spacing;
    end
    validateattributes(smoothing_mm,{'numeric'},{'scalar','real','finite','nonnegative'});
    kernel = 1;
    if smoothing_mm>0
        offsets = (-ceil(3*smoothing_mm/spacing):ceil(3*smoothing_mm/spacing))*spacing;
        kernel = exp(-0.5*(offsets/smoothing_mm).^2);
        kernel = kernel/sum(kernel);
    end
    previousHeight = cfg.thickness*ones(size(Xg));
    surfaceStats = zeros(height(manifest),5);
end
for k = 1:height(manifest)
    loaded = load(fullfile(source,manifest.Snapshot(k)),'frame');
    frame = loaded.frame;
    trueX = frame.currentXYZ_mm;
    trueTriangle = triangulation(faces,trueX(:,1:2));
    target = trueX-meta.referenceXYZ_mm;
    if strcmp(mode,'height')
        [id,bary] = pointLocation(trueTriangle,meta.referenceXYZ_mm(interiorTop,1:2));
        assert(all(isfinite(id)),'Height unavailable at one or more interior top nodes.');
        Z = reshape(trueX(faces(id,:),3),[],3);
        target = zeros(numel(nodes),3);
        target(interiorTop,3) = sum(bary.*Z,2)-cfg.thickness;
    end
    if surfaceMode
        % Extend only the region outside the clamped disk to the rim height.
        observed = frame.height_mm;
        if k==1
            footprint = isfinite(observed);
        end
        assert(all(isfinite(observed(footprint))),'Height data are missing inside the disk.');
        observed(~footprint) = cfg.thickness;
        started = tic;
        smoothedHeight = cfg.thickness+conv2(kernel,kernel,observed-cfg.thickness,'same');
        [u,R,stats] = palm.inverse.surfaceEquilibrium(model,u,nodes(interiorTop), ...
            meta.x_mm,meta.y_mm,previousHeight,smoothedHeight,k==height(manifest));
        elapsed = toc(started);
        previousHeight = smoothedHeight;
        iterations = stats.iterations;
        equilibriumResidual = stats.residual_N;
        surfaceStats(k,:) = [elapsed,stats.freeDofs,stats.positiveTangent, ...
            sum(max(0,R(topDofs(:,3)))),smoothing_mm];
        fprintf('  equilibrium %.3f s, %d iterations, residual %.3g N\n', ...
            elapsed,iterations,equilibriumResidual);
    else
        prescribed = target(interiorTop,:);
        [u,R,iterations] = palm.inverse.dirichlet(model,u,topDofs(:),prescribed(:));
        equilibriumResidual = norm(R(setdiff(model.free,topDofs)));
    end
    current = model.X+reshape(u,[],3);
    X = current(nodes,:);
    nodal = zeros(numel(nodes),3);
    nodal(interiorTop,:) = R(topDofs);

    % Reassemble exact reference nodal contact forces from archived quadrature.
    trueNodal = zeros(numel(nodes),3);
    for d = 1:3
        Q = (frame.currentArea_mm2.*frame.contactTraction_MPa(:,:,d).* ...
            meta.quadratureWeights')*meta.barycentric;
        trueNodal(:,d) = accumarray(faces(:),Q(:),[numel(nodes),1]);
    end
    assert(norm(trueNodal(~interiorTop,:),'fro') < 1e-10,'Contact touches clamped rim.');
    [pressure,traction,normal,area] = palm.inverse.traction(X,meta,M,nodal);
    [truthProjectedPressure,~,~,~] = palm.inverse.traction(trueX,meta,M,trueNodal);
    meanPressure = pressure*meta.quadratureWeights;
    truePressure = frame.contactPressure_MPa*meta.quadratureWeights;
    trueProjection = truthProjectedPressure*meta.quadratureWeights;
    threshold = max(.001,.05*max(truePressure));
    % Compare footprints at common spatial locations on the projected surface.
    idRef = pointLocation(trueTriangle,grid);
    inverseTri = triangulation(faces,X(:,1:2));
    [idInv,l] = pointLocation(inverseTri,grid);
    refActive = classify(idRef,truePressure,threshold);
    projActive = classify(idRef,trueProjection,threshold);
    invActive = classify(idInv,meanPressure,threshold);
    z = NaN(size(idInv));
    good = isfinite(idInv);
    zv = reshape(X(faces(idInv(good),:),3),[],3);
    z(good) = sum(l(good,:).*zv,2);
    roi = hypot(grid(:,1)-cfg.probeCenter(1),grid(:,2)-cfg.probeCenter(2))<=5;
    compare = isfinite(z) & isfinite(frame.height_mm(:)) & roi;
    force = sum(nodal,1);
    forceRef = sum(trueNodal,1);
    assert(abs(forceRef(3)+frame.force_N)<1e-9);
    w = area*meta.quadratureWeights';
    outward = sum(max(-pressure,0).*w,'all');
    tangential = traction+pressure.*reshape(normal,[],1,3);
    shear = sum(vecnorm(tangential,2,3).*w,'all');
    outside = sum(abs(pressure(truePressure<threshold,:)).*w(truePressure<threshold,:),'all');
    rows(k,:) = [manifest.Step(k),manifest.Travel_mm(k),manifest.CapDepth_mm(k), ...
        forceRef,force,norm(force),norm(forceRef),threshold*1000, ...
        nnz(refActive)*spacing^2,nnz(invActive)*spacing^2,overlap(refActive,invActive), ...
        overlap(projActive,invActive),sqrt(mean((z(compare)-frame.height_mm(compare)).^2)), ...
        outward,shear,outside,norm(nodal-trueNodal,'fro'), ...
        palm.fem.minimumJacobian(model,u),iterations, ...
        equilibriumResidual];
    assert(abs(sum(traction(:,:,3).*w,'all')-force(3))<1e-9);
    if strcmp(mode,'full')
        assert(norm(nodal-trueNodal,'fro')<1e-6,'Exact-displacement verification failed.');
    end
    series{k} = struct('XYZ_mm',X,'trueXYZ_mm',trueX,'pressure_MPa',pressure, ...
        'traction_MPa',traction,'truePressure_MPa',frame.contactPressure_MPa, ...
        'trueTraction_MPa',frame.contactTraction_MPa, ...
        'facePressure_MPa',meanPressure,'trueFacePressure_MPa',truePressure, ...
        'projectedTrueFacePressure_MPa',trueProjection, ...
        'height_mm',reshape(z,size(Xg)),'trueHeight_mm',frame.height_mm, ...
        'contactMask',reshape(invActive,size(Xg)),'trueContactMask',reshape(refActive,size(Xg)), ...
        'nodalForce_N',nodal,'trueNodalForce_N',trueNodal);
    fprintf('%s %s step %d, Fz %.6g vs %.6g N, IoU %.3f\n',caseName,mode, ...
        manifest.Step(k),-force(3),-forceRef(3),rows(k,15));
end
tableData = array2table(rows,'VariableNames',{'Step','Travel_mm','CapDepth_mm', ...
    'RefFx_N','RefFy_N','RefFz_N','Fx_N','Fy_N','Fz_N','ForceMagnitude_N','RefForceMagnitude_N', ...
    'Threshold_kPa','RefContactArea_mm2','ContactArea_mm2','ContactIoU','ProjectedRefIoU', ...
    'HeightRMSWithin5mm_mm','TensileIntegral_N','TangentialIntegral_N', ...
    'AbsNormalOutsideRefContact_N','NodalForceError_N','MinJ','NewtonIterations','Residual_N'});
if surfaceMode
    tableData = [tableData,array2table(surfaceStats,'VariableNames', ...
        {'Solve_s','FreeDofs','PositiveFinalTangent','UpwardNodalForce_N','Smoothing_mm'})];
end
writetable(tableData,fullfile(directory,'metrics.csv'));
save(fullfile(directory,'reconstruction.mat'),'meta','series','tableData');
end

function active = classify(id,p,threshold)
active = false(size(id));
good = isfinite(id);
active(good) = p(id(good)) >= threshold;
end

function value = overlap(a,b)
value = nnz(a&b)/max(1,nnz(a|b));
if ~any(a|b)
    value = 1;
end
end
