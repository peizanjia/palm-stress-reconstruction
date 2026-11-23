function run_generalization_samples(directory,worker,workers,batch)
% Generate deterministic, trajectory-separated indentation data.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'),fullfile(root,'config'));
if nargin<2
    worker = 1;
end
if nargin<3
    workers = 1;
end
validateattributes(workers,{'numeric'},{'scalar','integer','positive'});
validateattributes(worker,{'numeric'},{'scalar','integer','positive','<=',workers});
if nargin<4
    batch = 'factorial';
end
batch = validatestring(batch,{'factorial','combined','extension'});
if ~isfolder(directory)
    mkdir(directory);
end
rng(826,'twister');
splits = ["train","development","test","shape","position","tilt","deep","mesh"];
counts = [36,9,9,4,4,4,4,4];
combined = strcmp(batch,'combined');
extension = strcmp(batch,'extension');
if combined
    rng(2027,'twister');
    splits = "combined";
    counts = 6;
elseif extension
    rng(3011,'twister');
    splits = ["train_extension","development_extension"];
    counts = [12,6];
end
records = struct([]);
for block = 1:numel(splits)
    count = counts(block);
    L = zeros(count,9);
    for j = 1:size(L,2)
        L(:,j) = (randperm(count)'-rand(count,1))/count;
    end
    for i = 1:count
        r = L(i,:);
        family = mod(i-1,3)+1;
        long = 3.4+1.7*r(1);
        short = long*(0.58+0.25*r(2));
        axes = [long,short,3.1+1.4*r(3)];
        p = 2;
        if family==1
            axes(:) = long;
        elseif family==3
            p = 4;
        end
        yaw = 180*r(4);
        tilt = 18*r(5);
        radial = 6*sqrt(r(6));
        azimuth = 2*pi*r(7);
        maxDepth = 2.4+0.3*r(8);
        if combined
            p = 3;
            radial = 8+2*r(6);
            tilt = 23+9*r(5);
            yaw = 60+40*r(4);
            maxDepth = 3.12+0.35*r(8);
        elseif extension
            radial = 7.8*sqrt(r(6));
            tilt = 22*r(5);
            yaw = 140*r(4);
            yaw = yaw+40*(yaw>=60);
            maxDepth = 2.95+0.17*r(8);
        elseif block==4
            p = 3;
        elseif block==5
            radial = 8+2*r(6);
        elseif block==6
            axes = [long,short,3.1+1.4*r(3)];
            tilt = 23+9*r(5);
        elseif block==7
            maxDepth = 3.12+0.35*r(8);
        end
        rGeom = radial*[cos(azimuth),sin(azimuth)];
        d = [0.12,0.32,0.68,1.12,1.65,2.1,maxDepth];
        d(3:6) = d(3:6)+0.14*(r(9)-0.5);
        if block==7 || combined || extension
            d = [d(1:6),2.62,2.91,maxDepth];
        end
        entry = struct('split',splits(block),'id',splits(block)+"_"+sprintf('%02d',i), ...
            'axes',axes,'power',p,'yaw_deg',yaw,'tilt_deg',tilt,'center_mm',rGeom, ...
            'depths_mm',d,'mesh_mm',0.83);
        if block==8
            % Match earlier challenges at a finer spatial resolution.
            entry = records(sum(counts(1:i+2))+1);
            entry.id = "mesh_"+entry.id;
            entry.split = "mesh";
            entry.mesh_mm = 0.53;
        end
        records = [records;entry]; %#ok<AGROW>
    end
end
if worker==1
    save(fullfile(directory,'design.mat'),'records');
end
for k = worker:workers:numel(records)
    spec = records(k);
    destination = fullfile(directory,spec.id+".mat");
    if isfile(destination)
        continue
    end
    cfg = palm.indent.generalizationConfig(spec);
    writeCase(destination,cfg,spec);
end
end

function writeCase(destination,cfg,spec)
spacing = 0.25;
axis = -cfg.radius:spacing:cfg.radius;
[xx,yy] = meshgrid(axis,axis);
query = [xx(:),yy(:)];
n = numel(cfg.depths)+1;
heights = NaN([size(xx),n],'single');
diagnostics = NaN(n,8);
used = 0;
started = tic;
try
    solved = run_indentation(cfg,@record);
    status = solved.status;
catch errorData
    status = string(errorData.message);
end
heights = heights(:,:,1:used);
diagnostics = diagnostics(1:used,:);
travel = diagnostics(:,1);
force = diagnostics(:,4);
seconds = toc(started);
save(destination,'heights','travel','force','spec','status','diagnostics','axis','seconds');
fprintf('%s %d/%d, %.1f s, %s\n',spec.id,max(0,used-1),n-1,seconds,status);

    function record(step,model,state,data)
        x = model.X+reshape(state(1:3*model.n),[],3);
        tri = model.contactNodes;
        a = x(tri(:,2),1:2)-x(tri(:,1),1:2);
        b = x(tri(:,3),1:2)-x(tri(:,1),1:2);
        assert(all(a(:,1).*b(:,2)-a(:,2).*b(:,1)>0),'Top surface is not a height graph.');
        assert(abs(data(4)-data(5))<1e-6,'Reference force balance failed.');
        nodes = model.mesh.topNodes;
        [~,localTri] = ismember(tri,nodes);
        surface = triangulation(localTri,x(nodes,1:2));
        [face,weights] = pointLocation(surface,query);
        valid = isfinite(face);
        z = NaN(numel(face),1,'single');
        z(valid) = sum(weights(valid,:).*reshape(x(tri(face(valid),:),3),[],3),2);
        heights(:,:,step+1) = reshape(z,size(xx));
        diagnostics(step+1,:) = data;
        used = step+1;
    end
end
