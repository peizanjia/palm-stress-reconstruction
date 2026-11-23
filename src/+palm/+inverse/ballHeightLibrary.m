function library = ballHeightLibrary(directory)
% Read a fixed-center forward indentation series on its archived height grid.
saved = load(fullfile(directory,'surface_mesh.mat'),'meta');
meta = saved.meta;
manifest = readtable(fullfile(directory,'manifest.csv'),'TextType','string');
validateattributes(manifest.Travel_mm,{'numeric'},{'finite','increasing'});
assert(height(manifest)>=2,'At least two forward states are needed.');

[x,y] = meshgrid(meta.x_mm,meta.y_mm);
center = meta.config.probeCenter;
radius = hypot(x-center(1),y-center(2));
roi = radius<=5;
[~,centerSample] = min(radius(roi));
heights = zeros(nnz(roi),height(manifest));
for k = 1:height(manifest)
    saved = load(fullfile(directory,manifest.Snapshot(k)),'frame');
    heights(:,k) = saved.frame.height_mm(roi);
end
assert(all(isfinite(heights),'all'),'The 5 mm fitting region must have complete height coverage.');
library.roi = roi;
library.x_mm = meta.x_mm;
library.y_mm = meta.y_mm;
library.center_mm = center;
library.centerHeight_mm = heights(centerSample,:)';
library.heights_mm = heights;
library.travel_mm = manifest.Travel_mm;
library.force_N = manifest.Force_N;
library.step = manifest.Step;
end
