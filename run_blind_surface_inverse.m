function run_blind_surface_inverse(samples,output,freezeFile,worker,workers)
% Compare the full-order height-only inverse on every clean composite frame.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
if nargin<4, worker = 1; end
if nargin<5, workers = 1; end
validateattributes(workers,{'numeric'},{'scalar','integer','positive'});
validateattributes(worker,{'numeric'},{'scalar','integer','positive','<=',workers});
protocol = jsondecode(fileread(fullfile(root,'config','generalization_blind.json')));
sha = java.security.MessageDigest.getInstance('SHA-256');
sha.update(int8(fileread(freezeFile)));
token = lower(reshape(dec2hex(typecast(sha.digest(),'uint8'),2).',1,[]));
if ~isfolder(output), mkdir(output); end
for index = worker:workers:protocol.combined_trajectories
    trajectory = sprintf('blind_combined_%02d',index);
    data = load(fullfile(samples,[trajectory,'.mat']),'heights','axis','travel','force','freezeToken');
    assert(strcmp(data.freezeToken,token),'FEM comparison requires the frozen blind batch.');
    steps = protocol.combined_nonzero_steps;
    assert(numel(data.travel)==steps+1,'Incomplete FEM reference trajectory.');
    axis = data.axis;
    [x,y] = meshgrid(axis,axis);
    cfg = indentation_config();
    cfg.mesh.size = 0.83;
    cfg.mesh.fineRadius = 5.5;
    sigma = protocol.surface_inverse_smoothing_mm;
    spacing = axis(2)-axis(1);
    offsets = spacing*(-ceil(3*sigma/spacing):ceil(3*sigma/spacing));
    kernel = exp(-0.5*(offsets/sigma).^2);
    kernel = kernel/sum(kernel);
    flat = cfg.thickness*ones(size(x));
    rows = NaN(steps,6);
    status = strings(steps,1);
    for step = 1:steps
        started = tic;
        measured = double(data.heights(:,:,step+1));
        measured(~isfinite(measured)) = cfg.thickness;
        height = cfg.thickness+conv2(kernel,kernel,measured-cfg.thickness,'same');
        % The observed depression alone locates the mesh refinement.
        sag = max(0,cfg.thickness-height);
        weight = max(0,sag-0.85*max(sag,[],'all'));
        cfg.probeCenter = [sum(x.*weight,'all'),sum(y.*weight,'all')]/sum(weight,'all');
        estimate = NaN;
        residual = NaN;
        minJ = NaN;
        status(step) = "ok";
        try
            model = palm.indent.prepare(cfg);
            top = setdiff(model.mesh.topNodes,model.mesh.sideNodes);
            [u,reactions,stats] = palm.inverse.surfaceEquilibrium(model,zeros(3*model.n,1), ...
                top,axis,axis,flat,height,false);
            estimate = -sum(reactions(top+2*model.n));
            residual = stats.residual_N;
            minJ = palm.fem.minimumJacobian(model,u);
        catch problem
            status(step) = "failed: "+string(problem.message);
        end
        rows(step,:) = [data.travel(step+1),data.force(step+1),estimate,toc(started),residual,minJ];
        fprintf('%s step %d/%d %.2f s %s\n',trajectory,step,steps,rows(step,4),status(step));
    end
    tableData = array2table(rows,'VariableNames', ...
        {'Travel_mm','Reference_N','Estimate_N','Solve_s','Residual_N','MinJ'});
    tableData = addvars(tableData,repmat(string(trajectory),steps,1),'Before',1, ...
        'NewVariableNames','Trajectory');
    tableData.Status = status;
    writetable(tableData,fullfile(output,[trajectory,'.csv']));
end
end
