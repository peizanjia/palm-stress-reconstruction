function summary = summarize_indentation(directory, pattern)
% Compare 3-D indentation curves to Fig. 3c on a common measured depth range.
root = fileparts(mfilename('fullpath'));
if nargin < 1
    directory = fullfile(root,'results','indentation');
end
if nargin < 2
    pattern = 'coarse_H*.mat';
end
data = readtable(fullfile(root,'data','fig3c.csv'));
files = dir(fullfile(directory,pattern));
assert(~isempty(files),'No saved indentation cases matched the requested pattern.');
cases = cell(numel(files),1);
for k = 1:numel(files)
    loaded = load(fullfile(files(k).folder,files(k).name),'r');
    assert(height(loaded.r.history) >= 2, 'Too few converged points in %s.', files(k).name);
    cases{k} = loaded.r;
end
upper = min(cellfun(@(r) max(r.history.CapDepth_mm),cases));
common = data.CapDepth_mm <= upper;
assert(nnz(common) >= 2,'The cases have fewer than two common experimental observations.');
rows = NaN(numel(files),14);
status = strings(numel(files),1);
for k = 1:numel(files)
    r = cases{k};
    history = r.history;
    depth = [0;history.CapDepth_mm];
    [depth,indices] = unique(cummax(depth),'stable');
    force = [0;history.Force_N];
    force = force(indices);
    prediction = interp1(depth,force,data.CapDepth_mm,'pchip',NaN);
    observed = data.Force_N;
    assert(all(prediction(common)>0),'A comparison point has no positive computed contact force.');
    logRatio = log(prediction(common)./observed(common));
    scale = exp(-mean(logRatio));
    fitted = scale*prediction;
    covered = isfinite(prediction);
    rows(k,:) = [r.config.thickness,r.config.nu,r.config.mesh.size,r.config.penaltyFactor, ...
        max(depth),r.config.mu*1000*scale,exp(sqrt(mean(logRatio.^2))), ...
        exp(sqrt(mean((logRatio+log(scale)).^2))),sum(covered),sum(common), ...
        max(history.Penetration_mm),min(history.MinJ), ...
        max(abs(history.Force_N-history.Support_N)),r.nElements];
    status(k) = r.status;
    comparison = table(data.Force_N,data.CapDepth_mm,data.SD_mm,prediction,fitted,common, ...
        'VariableNames',{'ObservedForce_N','ObservedCapDepth_mm','ObservedSD_mm', ...
        'BaseForce_N','FittedForce_N','UsedInFit'});
    [~,name] = fileparts(files(k).name);
    writetable(history,fullfile(directory,name+"_history.csv"));
    writetable(comparison,fullfile(directory,name+"_comparison.csv"));
end
summary = array2table(rows,'VariableNames',{'Thickness_mm','Nu','MeshSize_mm','PenaltyFactor', ...
    'MaxCapDepth_mm','FittedMu_kPa','BaseForceRMSErrorFactor','FittedForceRMSErrorFactor', ...
    'CoveredObservations','CommonObservations','MaxPenetration_mm','MinJ', ...
    'MaxForceImbalance_N','Elements'});
summary.Status = status;
summary = sortrows(summary,{'Thickness_mm','MeshSize_mm','Nu','PenaltyFactor'});
end
