function [residual,K,diag] = assemble(model,cfg,state,depth,withTangent)
% Frictionless contact with the lower spherical surface, using a penalty.
u = state(1:3*model.n);
if model.nState > 3*model.n
    p = model.pressureScale*state(3*model.n+1:end);
    [internal,K,diag.minJ] = palm.fem.assemble(model,model.material,u,withTangent,p);
else
    [internal,K,diag.minJ] = palm.fem.assemble(model,model.material,u,withTangent);
end
residual = [];
if isempty(internal)
    return
end
x = model.X + reshape(u,[],3);
shaped = isfield(cfg,'probeShape');
if shaped
    [gap,normal,curvature] = palm.indent.shapedGap(model,cfg,x,depth,withTangent);
    upper = normal(:,:,3)>0;
else
    [gap,normal,rho,q] = palm.indent.probeGap(model,cfg,x,depth);
    upper = q(:,:,3)>0;
end
nTriangles = size(model.contactNodes,1);
f = zeros(nTriangles,9);
for i = 1:3
    cols = (i-1)*3+(1:3);
    f(:,cols) = (model.N'*(model.contactWeight.*gap.*normal(:,:,i)))';
end
contact = accumarray(model.contactDofs(:),f(:),[model.nState,1]);
residual = internal+contact;
diag.force = sum(f(:,7:9),'all');
fixedZ = model.fixed(model.fixed>2*model.n);
diag.reaction = sum(residual(fixedZ));
diag.upperContact = any(gap<0 & upper,'all');
if withTangent
    elementK = zeros(9,9,nTriangles);
    if ~shaped
        ratio = gap./rho;
    end
    for i = 1:3
        rows = (i-1)*3+(1:3);
        for j = 1:3
            cols = (j-1)*3+(1:3);
            if shaped
                term = normal(:,:,i).*normal(:,:,j)+gap.*curvature(:,:,i,j);
            else
                term = (1-ratio).*normal(:,:,i).*normal(:,:,j)+(i==j)*ratio;
            end
            integrated = model.products'*(model.contactWeight.*(gap<0).*term);
            elementK(rows,cols,:) = reshape(integrated,3,3,nTriangles);
        end
    end
    K = K + sparse(model.contactRows(:),model.contactCols(:),elementK(:),model.nState,model.nState);
end
end
