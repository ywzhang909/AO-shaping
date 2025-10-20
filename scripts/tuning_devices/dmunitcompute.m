clear;

% 计算100mm×100mm对应的像素尺寸  —（233，220）  （237，223）  
mm_size = 100; % 实际尺寸(mm)
resolution_for_70mm = 360; % 70mm对应的像素数
pixel_per_mm = resolution_for_70mm / 70; % 每毫米的像素数
pixel_size = round(mm_size * pixel_per_mm); % 100mm对应的像素数
mm_per_pixel = 1 / pixel_per_mm; % 每像素对应的毫米数 (约0.1944mm/像素)

% 读取stdWavefront下所有矩阵文件
folder_path = 'stdWavefront';
file_pattern = '*.txt';
file_list = dir(fullfile(folder_path, file_pattern));
zernike_base_matrix = zeros(360,360);
coeff = zeros(64, 1);
for i = 1:numel(file_list)
    file_path = fullfile(folder_path, file_list(i).name);
    A = load(file_path);
    A1 = reshape(A,360,360);
    zernike_base_matrix = zernike_base_matrix + A1 * coeff(i);
end

% 创建对应尺寸的二维数组
matrix_data = ones(pixel_size, pixel_size);

A = zernike_base_matrix;
A1 = reshape(A,360,360);

A_norm = normalize01(A1);
A_norm = A_norm-0.0;
% B_norm = normalize01(B1);
% B_norm = B_norm-0.7;
% C_norm = normalize01(C1);
% C_norm = C_norm-0.7;
% D_norm = normalize01(D1);
% D_norm = D_norm-0.7;

A_norm(A_norm < 0) = 0;
% B_norm(B_norm < 0) = 0;
% C_norm(C_norm < 0) = 0;
% D_norm(D_norm < 0) = 0;

% 计算各矩阵的圆域质心
[cx_A, cy_A] = centroidcaculation(A_norm);
% [cx_B, cy_B] = centroidcaculation(B_norm);
% [cx_C, cy_C] = centroidcaculation(C_norm);
% [cx_D, cy_D] = centroidcaculation(D_norm);

% 显示结果
fprintf('A1 质心: (%.2f, %.2f)\n', cx_A, cy_A);

figure;
imagesc(A1);
hold on;
plot(cx_A, cy_A, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
title('A1 及其质心');
axis image;
% fprintf('B1 质心: (%.2f, %.2f)\n', cx_B, cy_B);
% fprintf('C1 质心: (%.2f, %.2f)\n', cx_C, cy_C);
% fprintf('D1 质心: (%.2f, %.2f)\n', cx_D, cy_D);

% delta_x = (pixel_size-resolution_for_70mm)/2;
% delta_y = (pixel_size-resolution_for_70mm)/2;
% real_points_pixel = [
%     cx_A+delta_x, cy_A+delta_y;   % 点1
%     cx_B+delta_x, cy_B+delta_y;   % 点2
%     cx_C+delta_x, cy_C+delta_y;   % 点3
%     cx_D+delta_x, cy_D+delta_y];  % 点4
% 
% % 对于y轴，由于图像坐标系通常向下为正，而实际毫米坐标系向上为正，需要反转
% real_points_mm = real_points_pixel * mm_per_pixel;
% % real_points_mm(:, 2) = mm_size - real_points_mm(:, 2);  % 反转y轴，使左上角为(0,0)
% 
% % 定义要标注的点（mm坐标），左上角为(0,0)
% points_mm = [
%     50, 50-21.65;   % 点1
%     50, 50+21.65;   % 点2
%     50-21.65, 50;   % 点3
%     50+21.65, 50];  % 点4
% 
% % 将毫米坐标转换为像素坐标
% points_pixel = points_mm * pixel_per_mm;
% 
% tan_theta = (real_points_pixel(3,2)-real_points_pixel(4,2))/(real_points_pixel(3,1)-real_points_pixel(4,1));
% theta = atan(tan_theta);
% % 计算各矩阵的圆域质心
% [cx_A_2, cy_A_2] = calculateDerotation(cx_A, cy_A,theta);
% [cx_B_2, cy_B_2] = calculateDerotation(cx_B, cy_B,theta);
% [cx_C_2, cy_C_2] = calculateDerotation(cx_C, cy_C,theta);
% [cx_D_2, cy_D_2] = calculateDerotation(cx_D, cy_D,theta);
% 
% real_points_pixel_Derotation = [
%     cx_A_2+delta_y, cy_A_2+delta_y;   % 点1
%     cx_B_2+delta_y, cy_B_2+delta_y;   % 点2
%     cx_C_2+delta_y, cy_C_2+delta_y;   % 点3
%     cx_D_2+delta_y, cy_D_2+delta_y];  % 点4
% 
% delta_x = ((real_points_pixel_Derotation(1,1)-points_pixel(1,1))+(real_points_pixel_Derotation(2,1)-points_pixel(2,1))+(real_points_pixel_Derotation(3,1)-points_pixel(3,1))+(real_points_pixel_Derotation(4,1)-points_pixel(4,1)))/4*mm_per_pixel;
% delta_y = ((real_points_pixel_Derotation(1,2)-points_pixel(1,2))+(real_points_pixel_Derotation(2,2)-points_pixel(2,2))+(real_points_pixel_Derotation(3,2)-points_pixel(3,2))+(real_points_pixel_Derotation(4,2)-points_pixel(4,2)))/4*mm_per_pixel;
% 
% % 绘制二维数组
% figure('Name', '100mm×100mm二维数组可视化', 'Position', [100, 100, 700, 700]);
% imagesc(matrix_data);
% colormap(jet); % 使用jet颜色映射
% colorbar; % 显示颜色条
% title('对应100mm×100mm实际尺寸的二维数组', 'FontSize', 14);
% axis equal; % 保持横纵比例一致
% axis tight;
% 
% % 绘制点并添加标注
% hold on;  % 保持当前图像，以便添加点
% plot(points_pixel(1:4, 1), points_pixel(1:4, 2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
% 
% % 绘制点并添加标注
% hold on;  % 保持当前图像，以便添加点
% plot(real_points_pixel(1:4, 1), real_points_pixel(1:4, 2), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
% 
% % 绘制点并添加标注
% hold on;  % 保持当前图像，以便添加点
% plot(real_points_pixel_Derotation(1:4, 1), real_points_pixel_Derotation(1:4, 2), 'yo', 'MarkerSize', 8, 'LineWidth', 2);
% 
% % 添加尺寸标注
% annotation('textbox', [0.43, 0.02, 0.15, 0.05], ...
%     'String', sprintf('宽度: %d mm', mm_size), ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12);
% annotation('textbox', [0.02, 0.43, 0.1, 0.1], ...
%     'String', sprintf('高度: %d mm', mm_size), ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, ...
%     'Rotation', 90);
% 
% % 显示像素尺寸信息
% annotation('textbox', [0.4, 0.92, 0.2, 0.05], ...
%     'String', sprintf('像素尺寸: %d×%d', pixel_size, pixel_size), ...
%     'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 10);
% 
% % 显示网格以更清晰地看到矩阵结构
% grid on;
% set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.3);
% hold off;



% % 可选：在图像上标记质心
% figure;
% imagesc(A1);
% hold on;
% plot(cx_A, cy_A, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% title('A1 及其质心');
% axis image;
% 
% figure;
% imagesc(B1);
% hold on;
% plot(cx_B, cy_B, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% title('B1 及其质心');
% axis image;
% figure;
% imagesc(C1);
% hold on;
% plot(cx_C, cy_C, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% title('C1 及其质心');
% axis image;
% figure;
% imagesc(D1);
% hold on;
% plot(cx_D, cy_D, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% title('D1 及其质心');
% axis image;
