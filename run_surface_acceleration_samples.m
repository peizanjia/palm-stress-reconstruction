function run_surface_acceleration_samples(directory,batch)
% Generate whole trajectories for sampling and independent evaluation.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
assert(~isfolder(directory),'Use a new sample directory.');
mkdir(directory);
study = inverse_reference_config();
if nargin<2
    batch = 'independent';
end
batch = validatestring(batch,{'independent','multi_radius'});
if strcmp(batch,'independent')
    names = ["offset","larger_ball"];
    radii = [4,5];
    centers = [3,4;3,4];
else
    names = ["train_small","train_large","validation","new_shape"];
    radii = [3.5,5.5,4.5,4.75];
    centers = [-3,1;2,-4;0,0;-4,3];
end
depths = [0.12;0.32;0.62;0.92;1.32;1.72;2.12;2.52;2.92;3.12;3.32;3.52;3.72];
for i = 1:numel(names)
    cfg = study.indent;
    cfg.probeCenter = centers(i,:);
    cfg.probeRadius = radii(i);
    cfg.maxCapDepth = cfg.probeRadius;
    cfg.depths = depths(depths<radii(i));
    writeCase(names(i),cfg);
end

    function writeCase(name,cfg)
        travel = [0;cfg.depths];
        force = NaN(size(travel));
        heights = [];
        meta = [];
        trace = [];
        started = tic;
        result = run_indentation(cfg,@record);
        count = nnz(isfinite(force));
        heights = heights(:,:,1:count);
        travel = travel(1:count);
        force = force(1:count);
        status = result.status;
        save(fullfile(directory,name+".mat"),'heights','travel','force','cfg','status');
        fprintf('%s: %d/%d loaded frames in %.2f s, %s\n',name,count-1,numel(cfg.depths),toc(started),status);

        function record(step,model,u,diagnostics)
            if step==0
                [meta,trace] = palm.dataset.prepareSurface(model,cfg,study.heightSpacing);
                heights = NaN([trace.gridSize,numel(travel)]);
            end
            frame = palm.dataset.snapshot(model,cfg,meta,trace,u,diagnostics);
            heights(:,:,step+1) = frame.height_mm;
            force(step+1) = frame.force_N;
        end
    end
end
