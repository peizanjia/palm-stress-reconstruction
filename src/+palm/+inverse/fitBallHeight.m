function estimate = fitBallHeight(library,height_mm,samples)
% Fit a measured Eulerian height map to a fixed-center forward series.
% Optional sample indices let validation exclude measured states from the fit.
if nargin < 3
    samples = 1:numel(library.travel_mm);
end
validateattributes(samples,{'numeric'},{'vector','integer','increasing', ...
    '>=',1,'<=',numel(library.travel_mm)});
assert(numel(samples)>=2,'At least two forward states are needed.');
validateattributes(height_mm,{'numeric'},{'real','size',size(library.roi)});
observed = height_mm(library.roi);
assert(all(isfinite(observed)),'The fitting region must have complete height coverage.');

% Each adjacent pair defines a segment in height-map space. Project onto
% every segment and select the global least-squares fit on the polyline.
heights = library.heights_mm(:,samples);
start = heights(:,1:end-1);
change = diff(heights,1,2);
offset = observed-start;
lengthSquared = sum(change.^2,1);
assert(all(lengthSquared>0),'Adjacent forward states must have distinct height maps.');
fraction = min(1,max(0,sum(offset.*change,1)./lengthSquared));
residual = offset-change.*fraction;
[mse,segment] = min(mean(residual.^2,1));

left = samples(segment);
right = samples(segment+1);
alpha = fraction(segment);
estimate.travel_mm = (1-alpha)*library.travel_mm(left)+alpha*library.travel_mm(right);
estimate.force_N = (1-alpha)*library.force_N(left)+alpha*library.force_N(right);
estimate.heightRMS_mm = sqrt(mse);
estimate.atRangeLimit = (segment==1 && alpha==0) || ...
    (segment==numel(samples)-1 && alpha==1);
end
