% 清除变量和窗口
clear all; close all; clc;

% 读取神经网络数据
data = readmatrix('C:\Users\Joseph\Desktop\raybnn\neuron_train\data\neuron_graph_tensor.csv');

% 提取神经元和胶质细胞的坐标
neuron_pos = data(:, 1:3); % (X, Y, Z)
glia_pos = data(:, 4:6);   % (GliaX, GliaY, GliaZ)
WValues = data(:, 7);      % 连接权重
WRowIdx = uint32(data(:, 8)); % 连接起点
WColIdx = uint32(data(:, 9)); % 连接终点

% 归一化坐标
rs = max(neuron_pos, [], 'all') * 1.2;
neuron_pos = neuron_pos / rs;
glia_pos = glia_pos / rs;

% 计算连接权重范围
WValues_mean = mean(WValues);
WValues_std = std(WValues);
max_weight = WValues_mean + 2 * WValues_std;
min_weight = WValues_mean - 2 * WValues_std;

% 创建 Figure
figure; hold on;
colormap('jet');

% 绘制胶质细胞（红色）
scatter3(glia_pos(:,1), glia_pos(:,2), glia_pos(:,3), 150, 'r', 'filled');

% 绘制神经元（绿色）
scatter3(neuron_pos(:,1), neuron_pos(:,2), neuron_pos(:,3), 150, 'g', 'filled');

% 画出神经元连接（颜色代表连接权重）
cm = colormap;
for i = 1:20:size(WColIdx, 1)
    start = WColIdx(i);
    finish = WRowIdx(i);
    p1 = neuron_pos(start, :);
    p2 = neuron_pos(finish, :);
    weight = WValues(i);
    
    % 计算颜色
    [~, colorID] = min(abs(weight - linspace(min_weight, max_weight, size(cm, 1))));
    myColor = cm(colorID, :);
    
    % 画出连接
    dp = p2 - p1;
    quiver3(p1(1), p1(2), p1(3), dp(1), dp(2), dp(3), 0, 'Color', myColor, 'LineWidth', 2);
end

% 设定颜色条
set(gca, 'clim', [min_weight max_weight]);
c = colorbar;
ylabel(c, 'Weights', 'FontSize', 14);

% 设置 Figure 细节
grid on;
xlabel('X/r_s', 'FontSize', 14);
ylabel('Y/r_s', 'FontSize', 14);
zlabel('Z/r_s', 'FontSize', 14);
set(gca, 'FontSize', 12);
view(3);
hold off;

% 保存图片
exportgraphics(gcf, 'NeuronNetwork3D.png', 'Resolution', 600);
