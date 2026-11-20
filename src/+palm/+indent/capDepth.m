function [height,radius,penetration] = capDepth(model,cfg,u,depth)
% Area-equivalent radius of the current projected contact footprint.
n = 10;
x = model.X + reshape(u,[],3);
t = model.contactNodes;
center = [cfg.probeCenter,cfg.thickness+cfg.probeRadius-depth];
a = x(t(:,1),:)-center;
b = x(t(:,2),:)-center;
c = x(t(:,3),:)-center;
ab = b-a;
ac = c-a;
projected = abs(ab(:,1).*ac(:,2)-ab(:,2).*ac(:,1))/2;
closestVertex = min([vecnorm(a,2,2),vecnorm(b,2,2),vecnorm(c,2,2)],[],2);
near = closestVertex <= cfg.probeRadius + max(vecnorm(ab,2,2),vecnorm(ac,2,2));
a = a(near,:); ab = ab(near,:); ac = ac(near,:);
area = projected(near)/n^2;
gap = @(i,j) vecnorm(a+i/n*ab+j/n*ac,2,2)-cfg.probeRadius;
contactArea = 0;
penetration = 0;
for i = 0:n-1
    for j = 0:n-i-1
        g1 = gap(i,j);
        g2 = gap(i+1,j);
        g3 = gap(i,j+1);
        contactArea = contactArea + area'*cutFraction([g1,g2,g3]);
        penetration = max(penetration,-min([g1;g2;g3],[],'all'));
        if i+j < n-1
            g4 = gap(i+1,j+1);
            contactArea = contactArea + area'*cutFraction([g2,g3,g4]);
        end
    end
end
radius = sqrt(contactArea/pi);
height = cfg.probeRadius-sqrt(max(0,cfg.probeRadius^2-radius^2));
end

function fraction = cutFraction(g)
g = sort(g,2);
fraction = double(g(:,3)<=0 & g(:,1)<0);
one = g(:,1)<0 & g(:,2)>=0;
two = g(:,2)<0 & g(:,3)>0;
fraction(one) = g(one,1).^2./((g(one,1)-g(one,2)).*(g(one,1)-g(one,3)));
fraction(two) = 1-g(two,3).^2./((g(two,3)-g(two,2)).*(g(two,3)-g(two,1)));
end
