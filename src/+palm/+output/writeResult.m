function writeResult(result, directory)
% Preserve reference/current point correspondence and surface topology in MAT.
if ~isfolder(directory)
    mkdir(directory);
end
save(fullfile(directory, 'forward.mat'), 'result');
top = result.mesh.topNodes;
topData = [top, result.mesh.X(top,:), result.currentNodes(top,:), result.displacement(top,:)];
writetable(array2table(topData, 'VariableNames', ...
    {'Node', 'X_mm', 'Y_mm', 'Z_mm', 'x_mm', 'y_mm', 'z_mm', 'ux_mm', 'uy_mm', 'uz_mm'}), ...
    fullfile(directory, 'top_surface.csv'));
writetable(result.history, fullfile(directory, 'load_history.csv'));
end
