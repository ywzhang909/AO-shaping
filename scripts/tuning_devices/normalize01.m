function normalized = normalize01(matrix)
    min_val = min(matrix(:));
    max_val = max(matrix(:));
    % 避免除以零的情况
    if max_val == min_val
        normalized = zeros(size(matrix));
    else
        normalized = (matrix - min_val) / (max_val - min_val);
    end
end