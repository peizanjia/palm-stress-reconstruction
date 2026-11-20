function study = inverse_reference_config()
% Stage 01, task 03. Apply the selected 3-D model at common travel values.
study.indent = indentation_config();
study.indent.depths = (0.12:0.20:3.72)';
study.indent.maxCapDepth = study.indent.probeRadius;
study.caseNames = ["centered","eccentric"];
study.probeCenters = [0,0;6,0];   % mm from disk center
study.heightSpacing = 0.25;      % mm on the current x-y plane
end
