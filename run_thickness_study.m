function summary = run_thickness_study(thicknesses, cfg, label)
% Save each full 3-D case and compare optical cap depth with Fig. 3c.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'config'));
if nargin < 1
    thicknesses = [3.5,5,7,10,15,20];
end
if nargin < 2
    cfg = indentation_config();
    cfg.mesh.size = 1.05;
    cfg.penaltyFactor = 20;
    cfg.depths = (0.12:0.18:4.62)';
end
if nargin < 3
    label = "coarse";
end
directory = fullfile(root,'results','indentation');
if ~isfolder(directory)
    mkdir(directory);
end
for H = thicknesses(:)'
    cfg.thickness = H;
    r = run_indentation(cfg);
    path = fullfile(directory,sprintf('%s_H%g.mat',label,H));
    save(path,'r');
    fprintf('H = %g mm, %d accepted points, %s\n',H,height(r.history),r.status);
end
summary = summarize_indentation(directory,label+"_H*.mat");
writetable(summary,fullfile(directory,label+"_summary.csv"));
end
