function [P, J, invTranspose] = neoHookean(F, material)
% P = mu F + (lambda log J - mu) F^(-T), evaluated page by page.
cof = cat(2, cross(F(:,2,:), F(:,3,:), 1), ...
    cross(F(:,3,:), F(:,1,:), 1), cross(F(:,1,:), F(:,2,:), 1));
J = sum(F(:,1,:) .* cof(:,1,:), 1);
P = [];
invTranspose = [];
if all(isfinite(J) & J > 0, 'all')
    invTranspose = cof ./ J;
    P = material.mu*F + (material.lambda*log(J) - material.mu).*invTranspose;
end
end
