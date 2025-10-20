function[c_x, c_y] = centroidcaculation(matrix)
    % 获取矩阵尺寸
    [rows, cols] = size(matrix);
    % 创建坐标网格
    [x, y] = meshgrid(1:cols, 1:rows);
    % 计算总和
    sum_intensity = sum(matrix(:));
    % 计算质心坐标 (加权平均)
    c_x = sum(sum(matrix .* x)) / sum_intensity;
    c_y = sum(sum(matrix .* y)) / sum_intensity;
end