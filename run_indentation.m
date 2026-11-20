function result = run_indentation(cfg,onStep)
%RUN_INDENTATION Displacement-controlled, frictionless spherical indentation.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
if nargin == 0
    cfg = indentation_config();
end
if nargin < 2
    onStep = [];
end
positive = {'real','finite','scalar','positive'};
for name = ["radius","thickness","probeRadius","mu","penaltyFactor","minStep","forceTolerance","maxCapDepth"]
    validateattributes(cfg.(name),{'numeric'},positive);
end
for name = ["size","fineRadius","growth","outerSpacing"]
    validateattributes(cfg.mesh.(name),{'numeric'},positive);
end
validateattributes(cfg.mesh.growth,{'numeric'},{'>=',1});
if isfield(cfg,'pressureStabilization')
    validateattributes(cfg.pressureStabilization,{'numeric'},positive);
else
    validateattributes(cfg.nu,{'numeric'},{'real','scalar','>=',0,'<',0.5});
end
validateattributes(cfg.depths,{'numeric'},{'real','finite','vector','increasing','positive'});
validateattributes(cfg.probeCenter,{'numeric'},{'real','finite','size',[1,2]});
validateattributes(cfg.fixSide,{'logical'},{'scalar'});
validateattributes(cfg.maxIterations,{'numeric'},{'scalar','integer','positive'});
assert(norm(cfg.probeCenter) < cfg.radius && cfg.mesh.fineRadius <= cfg.radius, ...
    'The probe center and refinement radius must be inside the disk.');
assert(cfg.maxCapDepth <= cfg.probeRadius, 'Cap depth cannot exceed the hemisphere height.');
model = palm.indent.prepare(cfg);
result = palm.indent.advance(model,cfg,onStep);
result.config = cfg;
result.nElements = size(model.wedges,1);
end
