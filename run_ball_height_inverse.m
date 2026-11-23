function [summary,details] = run_ball_height_inverse()
% Replay the archived fixed-center cases and validate on held-out interior frames.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
reference = fullfile(root,'results','stage-01-forward-model', ...
    'task-03-inverse-reference','20261120_reference_v1');
baseline = fullfile(root,'results','stage-02-inverse', ...
    'task-01-dirichlet-reconstruction','20261120_inverse_v1');
cases = ["centered";"eccentric"];
metrics = zeros(numel(cases),13);
details = table();
for c = 1:numel(cases)
    source = fullfile(reference,cases(c));
    started = tic;
    library = palm.inverse.ballHeightLibrary(source);
    loadSeconds = toc(started);
    manifest = readtable(fullfile(source,'manifest.csv'),'TextType','string');
    previous = readtable(fullfile(baseline,cases(c),'height','metrics.csv'));
    assert(isequal(previous.Step,library.step),'Baseline frames do not match the forward library.');
    n = numel(library.step);
    data = zeros(n,5);
    for k = 1:n
        saved = load(fullfile(source,manifest.Snapshot(k)),'frame');
        samples = 1:n;
        if k>1 && k<n
            samples(k) = [];
        end
        estimate = palm.inverse.fitBallHeight(library,saved.frame.height_mm,samples);
        data(k,:) = [estimate.travel_mm,estimate.force_N,estimate.heightRMS_mm, ...
            estimate.atRangeLimit,0];
        if k==n-1
            fit = @() palm.inverse.fitBallHeight(library,saved.frame.height_mm,samples);
            data(k,5) = timeit(fit);
        end
    end
    checked = (2:n-1)';
    relative = 100*(data(:,2)-library.force_N)./library.force_N;
    oldRelative = 100*(-previous.Fz_N-library.force_N)./library.force_N;
    travelError = data(:,1)-library.travel_mm;
    forceError = data(:,2)-library.force_N;
    centerSlope = -diff(library.centerHeight_mm)./diff(library.travel_mm);
    assert(all(centerSlope>0),'Center height is not strictly decreasing along the library.');
    metrics(c,:) = [numel(checked),min(centerSlope),max(abs(travelError(checked))), ...
        mean(abs(relative(checked))),max(abs(relative(checked))), ...
        max(abs(forceError(checked))),mean(abs(oldRelative(checked))), ...
        max(abs(oldRelative(checked))),relative(n-1),oldRelative(n-1), ...
        abs(forceError(n)),loadSeconds,1000*data(n-1,5)];
    replay = true(n,1);
    replay(checked) = false;
    details = [details;table(repmat(cases(c),n,1),library.step,replay,library.travel_mm, ...
        data(:,1),library.force_N,data(:,2),relative,oldRelative,data(:,3),logical(data(:,4)), ...
        'VariableNames',{'Case','Step','EndpointReplay','ReferenceTravel_mm','FittedTravel_mm', ...
        'ReferenceForce_N','FittedForce_N','ForceError_pct','PreviousForceError_pct', ...
        'HeightRMS_mm','AtRangeLimit'})]; %#ok<AGROW>
end
summary = [table(cases,'VariableNames',{'Case'}),array2table(metrics,'VariableNames', ...
    {'HoldoutCount','MinCenterHeightSlope','MaxTravelError_mm','MeanAbsForceError_pct', ...
    'MaxAbsForceError_pct','MaxAbsForceError_N','PreviousMeanAbsForceError_pct', ...
    'PreviousMaxAbsForceError_pct','DeepHoldoutForceError_pct','PreviousDeepHoldoutForceError_pct', ...
    'TerminalReplayForceError_N','LibraryLoad_s','OnlineFit_ms'})];
disp(summary)
end
