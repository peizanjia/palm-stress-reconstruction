function directory = run_compressibility_case(runId,caseName)
%RUN_COMPRESSIBILITY_CASE Archive one full-3D center indentation comparison.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
runId = string(runId);
assert(isscalar(runId) && ~isempty(regexp(char(runId),'^[A-Za-z0-9][A-Za-z0-9_-]*$','once')), ...
    'Invalid run identifier.');
study = inverse_reference_config();
cfg = study.indent;
switch string(caseName)
    case "compressible045"
    case "nearly049"
        cfg.nu = 0.49;
    case "incompressible"
        cfg.pressureStabilization = 0.05;
    case "incompressible_coarse"
        cfg.mesh.size = 0.75;
        cfg.pressureStabilization = 0.05;
    otherwise
        error('Unknown comparison case.');
end
directory = fullfile(root,'results','stage-01-forward-model', ...
    'task-04-compressibility',runId,caseName);
assert(~isfolder(directory),'Use a new run identifier for repeated cases.');
mkdir(directory);
meta = [];
trace = [];
grids = [];
vol = zeros(numel(cfg.depths)+1,7);
started = tic;
r = run_indentation(cfg,@record);
seconds = toc(started);
summary = r.history;
summary = [summary,array2table(vol(2:height(summary)+1,:), ...
    'VariableNames',{'DeltaVolume_mm3','RMS_Jminus1','MinGaussJ','MaxGaussJ', ...
    'DeltaVolumeFromTop_mm3','TopRise_mm','TopDown_mm'})];
writetable(summary,fullfile(directory,'history.csv'));
save(fullfile(directory,'comparison.mat'),'r','meta','grids','seconds');
fprintf('%s finished %d/%d load steps in %.1f s. %s\n', ...
    caseName,height(summary),numel(cfg.depths),seconds,r.status);

    function record(step,model,state,diagnostics)
        if step == 0
            [meta,trace] = palm.dataset.prepareSurface(model,cfg,study.heightSpacing);
            grids = NaN([trace.gridSize,numel(cfg.depths)+1]);
        end
        frame = palm.dataset.snapshot(model,cfg,meta,trace,state,diagnostics);
        grids(:,:,step+1) = frame.height_mm;
        vz = reshape(frame.currentXYZ_mm(meta.triangles,3),[],3);
        projectedArea = frame.currentArea_mm2.*frame.outwardNormal(:,3);
        topVolume = sum(projectedArea.*(mean(vz,2)-cfg.thickness));
        stats = palm.fem.volumeStats(model,state(1:3*model.n));
        vol(step+1,:) = [stats,topVolume,max(frame.height_mm,[],'all','omitnan')-cfg.thickness, ...
            cfg.thickness-min(frame.height_mm,[],'all','omitnan')];
        fprintf('%s travel %.2f mm, F %.6g N, Delta V %.6g mm^3, RMS(J-1) %.6g\n', ...
            caseName,diagnostics(1),diagnostics(4),stats(1),stats(2));
    end
end
