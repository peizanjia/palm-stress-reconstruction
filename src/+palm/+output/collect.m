function result = collect(mesh, fem, material, u, f, history)
% Return coordinates, reaction forces and integration-point diagnostics.
result.mesh = mesh;
result.displacement = reshape(u, [], 3);
result.currentNodes = mesh.X + result.displacement;
result.history = history;
result.appliedNodalForce = reshape(f, [], 3);
[internal, ~, result.minGaussJ] = palm.fem.assemble(fem, material, u, false);
result.minJ = palm.fem.minimumJacobian(fem, u);
imbalance = internal - f;
reaction = zeros(size(f));
reaction(fem.fixed) = imbalance(fem.fixed);
result.reaction = reshape(reaction, [], 3);
result.appliedForce = sum(result.appliedNodalForce, 1);
result.supportForce = sum(result.reaction, 1);
result.forceImbalance = result.appliedForce + result.supportForce;
result.residualNorm = norm(imbalance(fem.free));
result.maxIndentation = max(-result.displacement(mesh.topNodes,3));

elementU = reshape(u(fem.dofs'), 6, 3, []);
F = eye(3) + pagemtimes(elementU, 'transpose', fem.centroidGrad, 'none');
[P, J] = palm.material.neoHookean(F, material);
assert(~isempty(P), 'Invalid deformation gradient at an element centroid.');
result.centroidJ = reshape(J, [], 1);
result.centroidCauchy = pagemtimes(P, 'none', F, 'transpose') ./ J;
end
