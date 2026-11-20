function summary = plot_dirichlet_inverse(runId,steps)
% Render pressure on 3-D facets and compare footprints and force curves.
root = fileparts(mfilename('fullpath'));
p = fullfile(root,'results','stage-02-inverse','task-01-dirichlet-reconstruction',runId);
figures = fullfile(p,'figures');
if ~isfolder(figures)
    mkdir(figures);
end
if nargin < 2
    steps = [10,19];
end
validateattributes(steps,{'numeric'},{'vector','integer','>=',1,'<=',19});
cases = ["centered","eccentric"];
names = ["Forward reference","Full displacement","Height only"];
colors = [23,105,170;160,75,0;133,84,162]/255;
fCurve = figure('Visible','off','Color','w','Position',[40,40,1300,860]);
layout = tiledlayout(fCurve,2,2,'TileSpacing','compact','Padding','compact');
title(layout,'Dirichlet inverse reconstruction | load–depth response');
summary = table();
for k = 1:numel(cases)
    source = fullfile(p,cases(k));
    f = load(fullfile(source,'full','reconstruction.mat'));
    h = load(fullfile(source,'height','reconstruction.mat'));
    ft = f.tableData;
    ht = h.tableData;
    assert(height(ft)==20 && height(ht)==20 && isequal(ft.RefFz_N,ht.RefFz_N));
    assert(max(ft.NodalForceError_N)<1e-6 && max(ft.HeightRMSWithin5mm_mm)<1e-10);
    assert(all(ft.MinJ>0 & ht.MinJ>0));
    label = strrep(cases(k),'_',' ');
    for col = 1:2
        ax = nexttile(layout,(k-1)*2+col);
        if col == 1
            depth = ft.CapDepth_mm;
            xLabel = 'Reference optical cap depth (mm)';
        else
            depth = ft.Travel_mm;
            xLabel = 'Prescribed probe travel (mm)';
        end
        hold(ax,'on');
        plot(ax,depth,-ft.RefFz_N,'-','Color',colors(1,:),'LineWidth',2.4, ...
            'DisplayName',names(1));
        plot(ax,depth,-ft.Fz_N,'o','Color',colors(2,:),'MarkerSize',5, ...
            'LineWidth',1.2,'DisplayName',names(2));
        plot(ax,depth,-ht.Fz_N,'--','Color',colors(3,:),'LineWidth',2.1, ...
            'DisplayName',names(3));
        xlabel(ax,xLabel);
        ylabel(ax,'Downward resultant (N)');
        title(ax,label);
        grid(ax,'on');
        legend(ax,'Location','northwest');
        box(ax,'off');
    end
    for j = (steps(:)+1)'
        plotSurfaces(figures,cases(k),f.meta,f.series{j},h.series{j},ft(j,:),ht(j,:),colors,names);
    end
    row = table(cases(k),ft.Travel_mm(end),ft.CapDepth_mm(end),-ft.RefFz_N(end), ...
        -ft.Fz_N(end),-ht.Fz_N(end),ht.ForceMagnitude_N(end), ...
        100*(ht.Fz_N(end)/ft.RefFz_N(end)-1),ft.RefContactArea_mm2(end), ...
        ft.ContactArea_mm2(end),ht.ContactArea_mm2(end),ft.ContactIoU(end),ht.ContactIoU(end), ...
        ht.TangentialIntegral_N(end),ht.TensileIntegral_N(end),ht.AbsNormalOutsideRefContact_N(end), ...
        max(ft.NodalForceError_N),ht.HeightRMSWithin5mm_mm(end), ...
        'VariableNames',{'Case','Travel_mm','CapDepth_mm','ReferenceDownwardForce_N', ...
        'FullDownwardForce_N','HeightDownwardForce_N','HeightResultantMagnitude_N','HeightForceError_pct', ...
        'ReferenceContactArea_mm2','FullContactArea_mm2','HeightContactArea_mm2', ...
        'FullIoU','HeightIoU','HeightTangentialIntegral_N','HeightTensileIntegral_N', ...
        'HeightNormalOutsideRefContact_N','MaxFullNodalForceError_N','HeightRMSWithin5mm_mm'});
    summary = [summary;row]; %#ok<AGROW>
end
saveFigure(fCurve,fullfile(figures,'load_depth'));
writetable(summary,fullfile(p,'summary.csv'));
end

function plotSurfaces(directory,name,meta,f,h,ft,ht,colors,names)
center = meta.config.probeCenter;
X = {f.trueXYZ_mm,f.XYZ_mm,h.XYZ_mm};
p = 1000*[f.trueFacePressure_MPa,f.facePressure_MPa,h.facePressure_MPa];
lim = max(abs(p(:)));
fFig = figure('Visible','off','Color','w','Position',[20,20,1480,900]);
tl = tiledlayout(fFig,2,3,'TileSpacing','compact','Padding','loose');
title(tl,sprintf('%s indentation | travel %.2f mm | optical cap %.2f mm', ...
    name,ft.Travel_mm,ft.CapDepth_mm));
for j = 1:3
    ax = nexttile(tl);
    surface(ax,meta.triangles,X{j},p(:,j),center,ft.Travel_mm,meta.config.thickness);
    colormap(ax,pressureMap());
    clim(ax,[-lim,lim]);
    title(ax,sprintf('%s  |  %.3f N',names(j),-verticalForce(j,ft,ht)));
    if j == 3
        cb = colorbar(ax,'eastoutside');
        cb.Label.String = 'Signed normal pressure (kPa)';
    end
end
ax = nexttile(tl);
hold(ax,'on');
drawBoundary(ax,meta,f.trueContactMask,colors(1,:),'-','Forward reference');
drawBoundary(ax,meta,f.contactMask,colors(2,:),':','Full displacement');
drawBoundary(ax,meta,h.contactMask,colors(3,:),'--','Height only');
axis(ax,'equal');
xlim(ax,center(1)+[-5,5]); ylim(ax,center(2)+[-5,5]);
xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)');
title(ax,sprintf('Projected contact | threshold %.1f kPa',ft.Threshold_kPa));
legend(ax,'Location','southoutside');
grid(ax,'on');

for j = 2:3
    ax = nexttile(tl);
    frame = f;
    stats = ft;
    if j == 3
        frame = h;
        stats = ht;
    end
    faces = meta.triangles;
    xyz = X{j};
    n = cross(xyz(faces(:,2),:)-xyz(faces(:,1),:),xyz(faces(:,3),:)-xyz(faces(:,1),:),2);
    n = n./vecnorm(n,2,2);
    t = frame.traction_MPa + frame.pressure_MPa.*reshape(n,[],1,3);
    shear = 1000*vecnorm(t,2,3)*meta.quadratureWeights;
    if j == 2
        firstAx = ax;
        maxShear = max(shear);
    else
        maxShear = max(maxShear,max(shear));
    end
    surface(ax,faces,xyz,shear,center,ft.Travel_mm,meta.config.thickness);
    colormap(ax,interp1([0,.45,1],[.98,.98,.98;.55,.70,.84;.10,.26,.47],linspace(0,1,256)));
    title(ax,sprintf('%s | shear integral %.3f N',names(j),stats.TangentialIntegral_N));
end
clim(firstAx,[0,max(maxShear,eps)]);
clim(ax,[0,max(maxShear,eps)]);
cb = colorbar(ax,'eastoutside');
cb.Label.String = 'Tangential traction (kPa)';
saveFigure(fFig,fullfile(directory,sprintf('%s_step_%03d_3d',name,ft.Step)));
end

function z = verticalForce(j,f,h)
values = [f.RefFz_N,f.Fz_N,h.Fz_N];
z = values(j);
end

function surface(ax,faces,X,values,center,travel,H)
patch(ax,'Faces',faces,'Vertices',X,'FaceVertexCData',values,'FaceColor','flat','EdgeColor','none');
axis(ax,'equal');
view(ax,[-38,48]);
xlim(ax,center(1)+[-5,5]); ylim(ax,center(2)+[-5,5]); zlim(ax,[H-travel-.2,H+.6]);
xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)'); zlabel(ax,'z (mm)');
grid(ax,'on');
end

function rgb = pressureMap()
rgb = interp1([-1,-.45,0,.45,1], ...
    [.11,.31,.56;.48,.66,.85;.96,.96,.96;.76,.49,.32;.49,.18,.09],linspace(-1,1,257));
end

function drawBoundary(ax,meta,mask,c,style,label)
contour(ax,meta.x_mm,meta.y_mm,double(mask),[.5,.5], ...
    'LineColor',c,'LineStyle',style,'LineWidth',2,'DisplayName',label);
end

function saveFigure(fig,path)
exportgraphics(fig,path+".png",'Resolution',170);
savefig(fig,path+".fig");
close(fig);
end
