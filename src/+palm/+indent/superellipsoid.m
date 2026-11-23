function shape = superellipsoid(axes,p,yaw_deg,tilt_deg)
% Convex rigid probe, positioned using its lowest point in the laboratory.
validateattributes(axes,{'numeric'},{'size',[1,3],'positive','finite'});
validateattributes(p,{'numeric'},{'scalar','finite','>=',2});
c = cosd(yaw_deg); s = sind(yaw_deg);
a = cosd(tilt_deg); b = sind(tilt_deg);
R = [c,-s,0;s,c,0;0,0,1]*[a,0,b;0,1,0;-b,0,a];
B = R./axes;
vertical = (R(3,:).*axes)';
dual = p/(p-1);
extreme = -sign(vertical).*abs(vertical/norm(vertical,dual)).^(dual-1);
shape.lowest = (R*(axes'.*extreme))';
shape.B = B;
shape.power = p;
shape.scale = 1/norm(B*(sign(extreme).*abs(extreme).^(p-1)));
end
