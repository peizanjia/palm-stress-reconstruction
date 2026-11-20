function [u, history] = equilibrium(fem, material, f, options)
% Load control with consistent Newton iterations and residual backtracking.
u = zeros(size(f));
free = fem.free;
fraction = 0;
initialStep = 1/options.initialSteps;
step = initialStep;
history = zeros(0,4);
while fraction < 1
    target = min(fraction + step, 1);
    rhs = target*f(free);
    tolerance = options.absoluteTolerance + options.relativeTolerance*norm(rhs);
    [success, trial, iterations, residual, minJ] = ...
        newton(u, rhs, tolerance, fem, material, options);
    if success
        u = trial;
        fraction = target;
        history(end+1,:) = [fraction, iterations, residual, minJ]; %#ok<AGROW>
        if options.verbose
            fprintf('Load %6.2f%%  Newton %2d  residual %.2e N  min J %.6f\n', ...
                100*fraction, iterations, residual, minJ);
        end
        step = min(initialStep, 1.5*step);
    else
        step = (target-fraction)/2;
        if step < options.minStep
            error('palm:NoEquilibrium', ...
                'Failed near load fraction %.6f (last converged %.6f, residual %.3g N).', ...
                target, fraction, residual);
        end
        if options.verbose
            fprintf('Retry from load fraction %.6f with increment %.6f\n', fraction, step);
        end
    end
end
history = array2table(history, ...
    'VariableNames', {'LoadFraction', 'NewtonIterations', 'Residual_N', 'MinGaussJ'});
end

function [success, u, iteration, residual, minJ] = newton(u, rhs, tolerance, fem, material, options)
free = fem.free;
success = false;
for iteration = 0:options.maxIterations
    [internal, tangent, minJ] = palm.fem.assemble(fem, material, u, true);
    r = internal(free) - rhs;
    residual = norm(r);
    if residual <= tolerance
        success = true;
        return
    end
    if iteration == options.maxIterations
        return
    end

    increment = -(tangent(free,free) \ r);
    if any(~isfinite(increment))
        return
    end
    accepted = false;
    for backtrack = 0:options.maxBacktracks
        alpha = 2^(-backtrack);
        candidate = u;
        candidate(free) = u(free) + alpha*increment;
        if palm.fem.minimumJacobian(fem, candidate) <= 0
            continue
        end
        candidateForce = palm.fem.assemble(fem, material, candidate, false);
        if ~isempty(candidateForce)
            candidateResidual = norm(candidateForce(free) - rhs);
            if candidateResidual <= (1 - 1e-4*alpha)*residual
                u = candidate;
                accepted = true;
                break
            end
        end
    end
    if ~accepted
        return
    end
end
end
