function values = volumeStats(fem,u)
% Reference-volume-weighted statistics; the quadrature integrates det F
% exactly for the straight wedges, independent of the material law.
elementU = reshape(u(fem.dofs'),6,3,[]);
reference = numel(fem.grad)*sum(fem.weight,'all');
current = 0;
defect = 0;
minJ = inf;
maxJ = -inf;
for k = 1:numel(fem.grad)
    F = eye(3) + pagemtimes(elementU,'transpose',fem.grad{k},'none');
    J = sum(F(:,1,:).*cross(F(:,2,:),F(:,3,:),1),1);
    current = current+sum(J.*fem.weight,'all');
    defect = defect+sum((J-1).^2.*fem.weight,'all');
    minJ = min(minJ,min(J,[],'all'));
    maxJ = max(maxJ,max(J,[],'all'));
end
values = [current-reference, sqrt(defect/reference),minJ,maxJ];
end
