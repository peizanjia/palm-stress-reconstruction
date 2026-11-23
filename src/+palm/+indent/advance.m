function result = advance(model,cfg,onStep)
% Follow equilibria with adaptive subdivision on residual line search failure.
u = zeros(model.nState,1);
depth = 0;
acceptedU = u;
previousU = u;
previousDepth = 0;
history = zeros(numel(cfg.depths),8);
n = 0;
maxCap = 0;
if ~isempty(onStep)
    onStep(0,model,u,[0,0,0,0,0,0,1,0]);
end
for target = cfg.depths(:)'
    if target >= cfg.thickness
        result = output(history,n,acceptedU,model,"probe would reach the rigid bottom");
        return
    end
    step = target-depth;
    while depth < target
        next = min(depth+step,target);
        guess = u;
        if depth > previousDepth
            predicted = u + (next-depth)/(depth-previousDepth)*(u-previousU);
            if palm.fem.minimumJacobian(model,predicted(1:3*model.n)) > 0
                guess = predicted;
            end
        end
        [ok,trial,diagnostics] = solveStep(model,cfg,guess,next);
        if ~ok && depth > previousDepth && any(guess ~= u)
            [ok,trial,diagnostics] = solveStep(model,cfg,u,next);
        end
        if ok
            previousDepth = depth;
            previousU = u;
            u = trial;
            depth = next;
            step = min(1.5*step,target-depth);
        else
            step = (next-depth)/2;
            if step < cfg.minStep
                result = output(history,n,acceptedU,model, ...
                    sprintf('Newton failed at travel %.5f mm near %.5f mm',next,depth));
                return
            end
        end
    end
    if diagnostics.upperContact
        message = "contact reached above the half-sphere";
        if isfield(cfg,'probeShape')
            message = "contact reached the upper probe surface";
        end
        result = output(history,n,acceptedU,model,message);
        return
    end
    acceptedU = u;
    if isfield(cfg,'probeShape')
        current = model.X+reshape(u(1:3*model.n),[],3);
        gap = palm.indent.shapedGap(model,cfg,current,depth,false);
        top = model.contactNodes;
        a = current(top(:,2),:)-current(top(:,1),:);
        b = current(top(:,3),:)-current(top(:,1),:);
        area = abs(a(:,1).*b(:,2)-a(:,2).*b(:,1))/2;
        radius = sqrt(sum((model.faceWeight'*(gap<0)).*area')/pi);
        penetration = -min(gap,[],'all');
        cap = NaN;
    else
        [cap,radius,penetration] = palm.indent.capDepth(model,cfg,u(1:3*model.n),depth);
    end
    n = n+1;
    history(n,:) = [depth,cap,radius,diagnostics.force,diagnostics.reaction, ...
        penetration,palm.fem.minimumJacobian(model,u(1:3*model.n)),diagnostics.residual];
    if ~isempty(onStep)
        onStep(n,model,u,history(n,:));
    end
    maxCap = max(cap,maxCap);
    if ~isfield(cfg,'probeShape') && maxCap >= cfg.maxCapDepth
        result = output(history,n,u,model,"cap-depth target reached");
        return
    end
end
result = output(history,n,u,model,"travel limit reached");
end

function [ok,u,diagnostics] = solveStep(model,cfg,u,depth)
free = model.free;
ok = false;
[r,~,diagnostics] = palm.indent.assemble(model,cfg,u,depth,false);
if isempty(r)
    return
end
for iteration = 0:cfg.maxIterations
    residual = norm(r(free));
    diagnostics.residual = residual;
    if residual < cfg.forceTolerance
        ok = true;
        return
    end
    if iteration == cfg.maxIterations
        return
    end
    [~,K] = palm.indent.assemble(model,cfg,u,depth,true);
    increment = -K(free,free)\r(free);
    if any(~isfinite(increment))
        return
    end
    improved = false;
    for b = 0:12
        scale = 2^(-b);
        trial = u;
        trial(free) = u(free) + scale*increment;
        if palm.fem.minimumJacobian(model,trial(1:3*model.n)) <= 0
            continue
        end
        [rTrial,~,trialDiagnostics] = palm.indent.assemble(model,cfg,trial,depth,false);
        if ~isempty(rTrial) && norm(rTrial(free)) <= (1-1e-4*scale)*residual
            u = trial;
            r = rTrial;
            diagnostics = trialDiagnostics;
            improved = true;
            break
        end
    end
    if ~improved
        return
    end
end
end

function result = output(history,n,u,model,message)
result.history = array2table(history(1:n,:), 'VariableNames', ...
    {'Travel_mm','CapDepth_mm','ContactRadius_mm','Force_N','Support_N', ...
     'Penetration_mm','MinJ','Residual_N'});
result.status = string(message);
result.displacement = reshape(u(1:3*model.n),[],3);
if model.nState > 3*model.n
    result.multiplier_MPa = model.pressureScale*u(3*model.n+1:end);
end
result.mesh = model.mesh;
end
