function [u,R,stats] = surfaceEquilibrium(model,u,nodes,x,y,previousHeight,height,checkTangent)
% Eliminate vertical surface DOFs using a measured Eulerian height graph.
n = model.n;
top = nodes(:)+n*(0:2);
free = setdiff(model.free,top(:,3));
[~,ix] = ismember(top(:,1),free);
[~,iy] = ismember(top(:,2),free);
m = numel(free);
identity = sparse(free,1:m,1,3*n,m);
spacing = min([diff(x(:));diff(y(:))]);
delta = spacing/100;
X = model.X(nodes,:);
stats.iterations = 0;
stats.substeps = 0;
stats.positiveTangent = NaN;
fraction = 0;
step = 1;
while fraction<1
    next = min(1,fraction+step);
    values = previousHeight+next*(height-previousHeight);
    F = griddedInterpolant({y(:),x(:)},values,'spline','none');
    [ok,trial,R,normResidual,iterations] = solve(u,F);
    stats.iterations = stats.iterations+iterations;
    if ok
        u = trial;
        fraction = next;
        stats.substeps = stats.substeps+1;
        step = min(1,1.5*step);
    else
        step = (next-fraction)/2;
        assert(step>=1/1024,'Surface equilibrium stalled at height fraction %.6g.',next);
    end
end
stats.residual_N = normResidual;
stats.freeDofs = m;
if checkTangent
    [~,~,~,K] = evaluate(u,F,true);
    order = symamd(K);
    [~,flag] = chol(K(order,order));
    stats.positiveTangent = flag==0;
end

    function [ok,state,r,normResidual,it] = solve(state,F)
        ok = false;
        [state,r,g,~,energy] = evaluate(state,F,false);
        normResidual = norm(g);
        ordering = [];
        for it = 0:40
            if isempty(g)
                return
            end
            if normResidual<2e-9
                ok = true;
                return
            end
            if it==40
                return
            end
            [~,~,~,K] = evaluate(state,F,true);
            if isempty(ordering)
                ordering = symamd(K);
            end
            reordered = K(ordering,ordering);
            [L,flag] = chol(reordered,'lower');
            if flag
                shift = max(1e-8,1e-5*max(abs(diag(reordered))));
                for regularize = 1:8
                    [L,flag] = chol(reordered+shift*speye(m),'lower');
                    if flag==0
                        break
                    end
                    shift = 10*shift;
                end
            end
            if flag
                return
            end
            direction = zeros(m,1);
            direction(ordering) = -(L'\(L\g(ordering)));
            directionalDerivative = g'*direction;
            improved = false;
            for backtrack = 0:12
                scale = 2^-backtrack;
                candidate = state;
                candidate(free) = state(free)+scale*direction;
                [candidate,rCandidate,gCandidate,~,candidateEnergy] = evaluate(candidate,F,false);
                tolerance = 16*eps(max(1,abs(energy)));
                energyDecrease = candidateEnergy<=energy+1e-4*scale*directionalDerivative;
                roundoff = abs(candidateEnergy-energy)<=tolerance && norm(gCandidate)<normResidual;
                if ~isempty(gCandidate) && (energyDecrease || roundoff)
                    state = candidate;
                    r = rCandidate;
                    g = gCandidate;
                    energy = candidateEnergy;
                    normResidual = norm(g);
                    improved = true;
                    break
                end
            end
            if ~improved
                return
            end
        end
    end

    function [state,r,g,K,energy] = evaluate(state,F,withTangent)
        r = [];
        g = [];
        K = [];
        energy = inf;
        xx = X(:,1)+state(top(:,1));
        yy = X(:,2)+state(top(:,2));
        h = F(yy,xx);
        dxp = F(yy,xx+delta);
        dxm = F(yy,xx-delta);
        dyp = F(yy+delta,xx);
        dym = F(yy-delta,xx);
        hx = (dxp-dxm)/(2*delta);
        hy = (dyp-dym)/(2*delta);
        if any(~isfinite([h;hx;hy]))
            return
        end
        state(top(:,3)) = h-X(:,3);
        if palm.fem.minimumJacobian(model,state)<=0
            return
        end
        [r,Kfull,~,energy] = palm.fem.assemble(model,model.material,state,withTangent);
        % Virtual displacement along the graph: du_z = h_x du_x + h_y du_y.
        T = identity+sparse([top(:,3);top(:,3)],[ix;iy],[hx;hy],3*n,m);
        g = T'*r;
        if withTangent
            hxx = (dxp-2*h+dxm)/delta^2;
            hyy = (dyp-2*h+dym)/delta^2;
            hxy = (F(yy+delta,xx+delta)-F(yy-delta,xx+delta) ...
                -F(yy+delta,xx-delta)+F(yy-delta,xx-delta))/(4*delta^2);
            rz = r(top(:,3));
            curvature = sparse([ix;iy;ix;iy],[ix;iy;iy;ix], ...
                [rz.*hxx;rz.*hyy;rz.*hxy;rz.*hxy],m,m);
            K = T'*Kfull*T+curvature;
        end
    end
end
