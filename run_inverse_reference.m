function directory = run_inverse_reference(runId,caseNames)
%RUN_INVERSE_REFERENCE Archive stage 01/task 03 synthetic inverse benchmarks.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
study = inverse_reference_config();
if nargin < 1 || strlength(string(runId)) == 0
    runId = string(datetime('now','TimeZone','Asia/Singapore','Format','yyyyMMdd_HHmmss_SSS'));
end
runId = string(runId);
assert(isscalar(runId) && ~isempty(regexp(char(runId),'^[A-Za-z0-9][A-Za-z0-9_-]*$','once')), ...
    'Use letters, digits, hyphens and underscores in the run identifier.');
if nargin < 2
    caseNames = study.caseNames;
end
[known,indices] = ismember(string(caseNames),study.caseNames);
assert(~isempty(indices) && all(known,'all') && numel(unique(indices)) == numel(indices), ...
    'Select centered, eccentric, or both cases.');
directory = fullfile(root,'results','stage-01-forward-model','task-03-inverse-reference',runId);
for k = indices(:)'
    cfg = study.indent;
    cfg.probeCenter = study.probeCenters(k,:);
    runCase(fullfile(directory,study.caseNames(k)),cfg,study.heightSpacing);
end
end

function runCase(directory,cfg,spacing)
assert(~isfolder(directory),'Case directory already exists. Use a new run identifier.');
snapshotDir = fullfile(directory,'snapshots');
[created,message] = mkdir(snapshotDir);
assert(created,'%s',message);
meta = [];
trace = [];
steps = [0;cfg.depths(:)];
data = zeros(numel(steps),8);
files = strings(numel(steps),1);
result = run_indentation(cfg,@writeStep);
assert(height(result.history) == numel(cfg.depths), ...
    'Incomplete sequence. Valid individual snapshots remain in %s (%s).',directory,result.status);
manifest = array2table(data,'VariableNames',result.history.Properties.VariableNames);
manifest = addvars(manifest,(0:numel(steps)-1)',files,'Before',1, ...
    'NewVariableNames',{'Step','Snapshot'});
writetable(manifest,fullfile(directory,'manifest.csv'));

    function writeStep(step,model,u,diagnostics)
        if step == 0
            [meta,trace] = palm.dataset.prepareSurface(model,cfg,spacing);
            save(fullfile(directory,'surface_mesh.mat'),'meta');
        end
        frame = palm.dataset.snapshot(model,cfg,meta,trace,u,diagnostics);
        assert(abs(frame.contactForce_N(3)+diagnostics(4)) < 1e-9, ...
            'Exported contact traction is inconsistent with the solved force.');
        name = sprintf('step_%03d.mat',step);
        save(fullfile(snapshotDir,name),'frame');
        data(step+1,:) = diagnostics;
        files(step+1) = "snapshots/"+name;
        fprintf('%s step %02d, travel %.2f mm, force %.6g N\n', ...
            directory,step,diagnostics(1),diagnostics(4));
    end
end
