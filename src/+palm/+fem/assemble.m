function [internal, tangent, minJ] = assemble(fem, material, u, withTangent, pressure)
% Total-Lagrangian residual and the consistent derivative on the reference mesh.
elementU = reshape(u(fem.dofs'), 6, 3, []);
nElements = size(elementU, 3);
mixed = nargin == 5;
if mixed
    elementPressure = reshape(pressure(fem.wedges'),6,1,nElements);
    constraint = zeros(6,1,nElements);
    if withTangent
        coupling = zeros(18,6,nElements);
    end
end
elementForce = zeros(6, 3, nElements);
tangent = [];
internal = [];
minJ = inf;
if withTangent
    elementTangent = zeros(18, 18, nElements);
end
for q = 1:numel(fem.grad)
    G = fem.grad{q};
    F = eye(3) + pagemtimes(elementU, 'transpose', G, 'none');
    [P, J, invTranspose] = palm.material.neoHookean(F, material);
    minJ = min(minJ, min(J, [], 'all'));
    if isempty(P)
        return
    end
    c = material.lambda*log(J) - material.mu;
    volumetricDerivative = material.lambda;
    if mixed
        N = fem.pressureBasis(:,q);
        pAtQuad = pagemtimes(N','none',elementPressure,'none');
        volumetricDerivative = pAtQuad.*J;
        c = volumetricDerivative-material.mu;
        P = material.mu*F + c.*invTranspose;
        constraint = constraint + N.*(fem.pressureScale*(J-1).*fem.weight);
    end
    elementForce = elementForce + ...
        pagemtimes(G, 'none', P, 'transpose') .* fem.weight;
    if withTangent
        gInv = pagemtimes(G, 'none', invTranspose, 'transpose');
        base = material.mu * pagemtimes(G, 'none', G, 'transpose');
        if mixed
            dJ = reshape(gInv.*J,18,1,nElements);
            coupling = coupling + pagemtimes(dJ,N').*(fem.pressureScale*fem.weight);
        end
        for i = 1:3
            rows = (i-1)*6 + (1:6);
            for k = 1:3
                cols = (k-1)*6 + (1:6);
                first = pagemtimes(gInv(:,i,:), 'none', gInv(:,k,:), 'transpose');
                second = pagemtimes(gInv(:,k,:), 'none', gInv(:,i,:), 'transpose');
                block = volumetricDerivative.*first - c.*second + (i == k)*base;
                elementTangent(rows,cols,:) = elementTangent(rows,cols,:) + block.*fem.weight;
            end
        end
    end
end
ndof = 3*fem.nNodes;
internal = accumarray(reshape(fem.dofs', [], 1), elementForce(:), [ndof,1]);
if mixed
    constraint = accumarray(reshape(fem.wedges',[],1),constraint(:),[fem.nNodes,1]);
    scaledPressure = pressure/fem.pressureScale;
    internal = [internal;constraint-fem.pressureStabilization*scaledPressure];
end
if withTangent
    tangent = sparse(fem.rows(:), fem.cols(:), elementTangent(:), ndof, ndof);
    if mixed
        B = sparse(fem.couplingRows(:),fem.couplingCols(:),coupling(:),ndof,fem.nNodes);
        tangent = [tangent,B;B',-fem.pressureStabilization];
    end
end
end
