function summary = summarize_compressibility(runId)
% Inspect archived full-3D comparison cases at the same prescribed travel.
root = fileparts(mfilename('fullpath'));
directory = fullfile(root,'results','stage-01-forward-model','task-04-compressibility',runId);
names = ["compressible045","nearly049","incompressible","incompressible_coarse"];
archive = cell(numel(names),1);
history = archive;
for k = 1:numel(names)
    path = fullfile(directory,names(k));
    archive{k} = load(fullfile(path,'comparison.mat'),'r','meta','grids','seconds');
    history{k} = readtable(fullfile(path,'history.csv'));
    h = history{k};
    assert(height(h) >= 1);
    assert(all(h.MinJ>0));
    assert(max(abs(h.Force_N-h.Support_N)) < 1e-6);
    assert(max(abs(h.DeltaVolumeFromTop_mm3-h.DeltaVolume_mm3)) < 1e-7);
    if k > 1
        assert(isequal(archive{k}.meta.x_mm,archive{1}.meta.x_mm) && ...
            isequal(archive{k}.meta.y_mm,archive{1}.meta.y_mm));
    end
end
assert(isequal(archive{1}.r.mesh,archive{2}.r.mesh,archive{3}.r.mesh), ...
    'The fine cases must use exactly the same reference mesh.');

nFine = min(cellfun(@height,history(1:3)));
[x,y] = meshgrid(archive{1}.meta.x_mm,archive{1}.meta.y_mm);
rows = zeros(numel(names),17);
for k = 1:numel(names)
    a = archive{k};
    h = history{k};
    nCommon = min(nFine,height(h));
    assert(isequal(h.Travel_mm(1:nCommon),history{1}.Travel_mm(1:nCommon)));
    at = h(nCommon,:);
    dz = a.grids(:,:,nCommon+1)-archive{1}.grids(:,:,nCommon+1);
    local = isfinite(dz) & hypot(x,y)<=5;
    v0 = sum(a.meta.referenceArea_mm2)*a.meta.config.thickness;
    if isfield(a.r,'multiplier_MPa')
        dofs = 4*size(a.r.mesh.X,1);
    else
        dofs = 3*size(a.r.mesh.X,1);
    end
    rows(k,:) = [height(h),dofs,at.Travel_mm,at.Force_N,at.CapDepth_mm, ...
        at.TopRise_mm,at.TopDown_mm,at.DeltaVolume_mm3,100*at.DeltaVolume_mm3/v0, ...
        at.RMS_Jminus1,at.MinGaussJ,at.MaxGaussJ,sqrt(mean(dz(local).^2)), ...
        max(abs(at.DeltaVolumeFromTop_mm3-at.DeltaVolume_mm3)), ...
        max(abs(h.Force_N-h.Support_N)),at.Penetration_mm,a.seconds];
end
summary = array2table(rows,'VariableNames',{'CompletedSteps','DOFsIncludingFixed','CommonTravel_mm', ...
    'Force_N','CapDepth_mm','TopRise_mm','TopDown_mm','DeltaVolume_mm3','DeltaVolumePercent', ...
    'RMS_Jminus1','MinGaussJ','MaxGaussJ','HeightRMSvs045Within5mm_mm','VolumeCheckError_mm3', ...
    'MaxForceImbalance_N','Penetration_mm','WallSeconds'});
summary = addvars(summary,names','Before',1,'NewVariableNames','Case');
writetable(summary,fullfile(directory,'summary.csv'));
end
