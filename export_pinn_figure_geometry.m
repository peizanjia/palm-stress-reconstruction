function export_pinn_figure_geometry(sampleFile,destination)
% Reproduce one complete archived trajectory and export its terminal mesh.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
archive = load(sampleFile,'spec','travel','force');
spec = archive.spec;
cfg = palm.indent.generalizationConfig(spec);
result = run_indentation(cfg);
assert(isequal(archive.travel(2:end),result.history.Travel_mm), ...
    'Reproduced and archived trajectories have different travel steps.');
replayError_N = max(abs(result.history.Force_N-archive.force(2:end)));
assert(replayError_N<1e-6,'Reproduced force disagrees with the archive.');
reference = result.mesh.X;
deformed = reference+result.displacement;
top = result.mesh.top;
wedges = result.mesh.wedges;
height_mm = cfg.thickness;
travel_mm = result.history.Travel_mm(end);
force_N = result.history.Force_N(end);
rotation = cfg.probeShape.B.*spec.axes;
probeOrigin = [spec.center_mm,height_mm-travel_mm]-cfg.probeShape.lowest;
save(destination,'spec','reference','deformed','top','wedges','height_mm', ...
    'travel_mm','force_N','rotation','probeOrigin','replayError_N','-v7');
fprintf('Exported %s, terminal force %.8g N, max replay deviation %.3g N\n', ...
    spec.id,force_N,replayError_N);
end
