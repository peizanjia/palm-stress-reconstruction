function [u,residual,nIterations] = dirichlet(model,u,dofs,target)
% Nonlinear displacement-controlled solve. Only interior DOFs equilibrate.
free = setdiff(model.free,dofs);
start = u(dofs);
fraction = 0;
step = 1;
nIterations = 0;
while fraction < 1
    next = min(1,fraction+step);
    trial = u;
    trial(dofs) = start + next*(target-start);
    [ok,trial,residual,it] = solve(model,free,trial);
    nIterations = nIterations+it;
    if ok
        fraction = next;
        u = trial;
        step = min(1,1.5*step);
    else
        step = (next-fraction)/2;
        assert(step >= 1/1024,'Dirichlet solve stalled near fraction %.6g.',next);
    end
end
end

function [ok,u,r,it] = solve(model,free,u)
ok = false;
r = [];
for it = 0:24
    if palm.fem.minimumJacobian(model,u)<=0
        return
    end
    r = palm.fem.assemble(model,model.material,u,false);
    errorNorm = norm(r(free));
    if errorNorm < 2e-10
        ok = true;
        return
    end
    if it == 24
        return
    end
    [~,K] = palm.fem.assemble(model,model.material,u,true);
    du = -K(free,free)\r(free);
    improved = false;
    for b = 0:12
        candidate = u;
        scale = 2^(-b);
        candidate(free) = u(free)+scale*du;
        if palm.fem.minimumJacobian(model,candidate) <= 0
            continue
        end
        rc = palm.fem.assemble(model,model.material,candidate,false);
        if ~isempty(rc) && norm(rc(free)) <= (1-1e-4*scale)*errorNorm
            u = candidate;
            improved = true;
            break
        end
    end
    if ~improved
        return
    end
end
end
