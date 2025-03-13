clear all; close all;
fontsz = 16;

% 读取数据
initial_size = readmatrix('initial_cell_num.csv');
computation_time = readmatrix('collision_run_time.csv');
computation_time_batch = readmatrix('collision_run_time_batch.csv');
computation_time_serial = readmatrix('collision_run_time_serial.csv');

% 计算均值 & 误差
computation_time_mean = mean(reshape(computation_time, [], 10), 2);
computation_time_std = std(reshape(computation_time, [], 10), 0, 2);
computation_time_batch_mean = mean(reshape(computation_time_batch, [], 10), 2);
computation_time_batch_std = std(reshape(computation_time_batch, [], 10), 0, 2);
computation_time_serial_mean = mean(reshape(computation_time_serial, [], 10), 2);
computation_time_serial_std = std(reshape(computation_time_serial, [], 10), 0, 2);

% 计算拟合曲线
fit_serial = polyfit(log(initial_size), log(computation_time_mean), 1);
fit_batch = polyfit(log(initial_size(1:length(computation_time_batch_mean))), log(computation_time_batch_mean), 1);
fit_mini_batch = polyfit(log(initial_size(1:length(computation_time_serial_mean))), log(computation_time_serial_mean), 1);

% 绘制误差条
figure;
hold on;
errorbar(initial_size, computation_time_mean, computation_time_std, '.r', 'LineWidth', 2);
plot(initial_size, exp(polyval(fit_serial, log(initial_size))), 'r-', 'LineWidth', 2);
errorbar(initial_size(1:length(computation_time_batch_mean)), computation_time_batch_mean, computation_time_batch_std, '.g', 'LineWidth', 2);
plot(initial_size(1:length(computation_time_batch_mean)), exp(polyval(fit_batch, log(initial_size(1:length(computation_time_batch_mean))))), 'g-', 'LineWidth', 2);
errorbar(initial_size(1:length(computation_time_serial_mean)), computation_time_serial_mean, computation_time_serial_std, '.b', 'LineWidth', 2);
plot(initial_size(1:length(computation_time_serial_mean)), exp(polyval(fit_mini_batch, log(initial_size(1:length(computation_time_serial_mean))))), 'b-', 'LineWidth', 2);

% 设置 log-log 轴
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', fontsz);
xlabel('Number of Cells', 'FontSize', fontsz);
ylabel('Collision Computation Time (s)', 'FontSize', fontsz);
legend({'Serial', 'Batch', 'Mini Batch'}, 'FontSize', fontsz-2, 'Location', 'SouthEast', 'Box', 'off');
hold off;
drawnow;

% 保存图片
saveas(gcf, 'collision_runtime.png');
