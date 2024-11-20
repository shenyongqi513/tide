close all;
clear;
 clc;
%% 导入潮汐数据
% 第一个数对应的时间为2003年3月3日0时，相邻数据的时间间隔为1个小时。
tide_arr = importdata('6.txt');    %此处为导入本地潮汐数据文件
 
%% 根据已知数据绘出潮流观测曲线
x1 = 1:721;              %x为观测矩阵中第一列，既数据初始编号
y1 = tide_arr(:,2);      %y为观测矩阵中第二列，既观测潮位
plot(x1,y1);
xlabel('序列号');
ylabel('潮位（cm）');
title('观测潮位与自报潮位');
hold on


%% 选取8个主要分潮O1,P1,K1,M2,K2,Q1,S2,N2
% 查表得八个主要分潮的杜德森数
% 数组中分别对应u1,u2,u3,u4,u5,u6,u0
O1_du = [1,-1,0,0,0,0,-1];
P1_du = [1,1,-2,0,0,0,-1];
K1_du = [1,1,0,0,0,0,1];
M2_du = [2,0,0,0,0,0,0];
K2_du = [2,2,0,0,0,0,0];
Q1_du = [1,-2,0,1,0,0,-1];
S2_du = [2,2,-2,0,0,0,0];
N2_du = [2,-1,0,1,0,0,0];


%% 计算时间原点
% 设置时间的起点
year = 2003;
month = 3;
day = 3;
hour = 0;

% 计算具体的时间
start_time = datetime(year, month, day, hour, 0, 0);

% 如果需要一个时间间隔，每次增加1小时
time_interval = hours(1);

% 输出时间
disp(['计算出的起始时间是: ', char(start_time)]);

% 如果要查看每一小时的时间，可以这样做：
N = 721; % 假设我们有721个数据点，每个代表一个小时
time_series = start_time + (0:N-1) * time_interval;

% 显示计算得到的时间序列的一部分
disp('前5个时间点：');
disp(time_series(1:5));

%计算出中间时刻对应的时间并输出
 
 
%% 计算基本天文元素随时间的变化速度
omega_t =(2*pi) / (24 + (50/60));         %平太阴时角，周期一个平太阴日=24小时50分钟
omega_s = (2*pi) / (27.32*24);            %月球平均经度，周期一个回归月 = 27.32个平太阳日
omega_h = (2*pi) / (365.2422*24);         %太阳平均经度，周期一个回归年 = 265.2422个平太阳日
omega_p1 = (2*pi) / (8.85*365.2422*24);   %月球近地点平均经度，周期为8.85年
omega_n = (2*pi) / (18.61*365.2422*24);   %白道升交点在黄道上经度的负值，周期为18.61年
omega_p2 = (2*pi) / (21000*365.2422*24);  %太阳近地点平均经度，周期为21000年，u6=0此项可以忽略
 
%基本天文分潮随时间变化率矩阵
basic_value_omega = [omega_t,omega_s,omega_h,omega_p1,omega_n,omega_p2]';
 
%% 分别计算八个分潮的角速率
omega_O1 = O1_du(1:6) * basic_value_omega;
omega_P1 = P1_du(1:6) * basic_value_omega;
omega_K1 = K1_du(1:6) * basic_value_omega;
omega_M2 = M2_du(1:6) * basic_value_omega;
omega_K2 = K2_du(1:6) * basic_value_omega;
omega_Q1 = Q1_du(1:6) * basic_value_omega;
omega_S2 = S2_du(1:6) * basic_value_omega;
omega_N2 = N2_du(1:6) * basic_value_omega;
% 创建八个分潮角速度向量
Omega = [omega_O1,omega_P1,omega_K1,omega_M2,omega_K2,omega_Q1,omega_S2,omega_N2];
 
fprintf('O1分潮的角速率：%frad/h\n',omega_O1);
fprintf('P1分潮的角速率：%frad/h\n',omega_P1);
fprintf('K1分潮的角速率：%frad/h\n',omega_K1);
fprintf('M2分潮的角速率：%frad/h\n',omega_M2);
fprintf('K2分潮的角速率：%frad/h\n',omega_K2);
fprintf('Q1分潮的角速率：%frad/h\n',omega_Q1);
fprintf('S2分潮的角速率：%frad/h\n',omega_S2);
fprintf('N2分潮的角速率：%frad/h\n',omega_N2);
 

 
% 计算天文元素（以弧度制计算）
Y = 2003; M = 3; D = 3; t = 0;
n_base = D + 58; % 重新命名变量
i = fix((Y - 1900) / 4);
deg2rad = pi / 180; % 角度转弧度的常数

% 月球平均经度s
s = deg2rad * (277.02 + 129.3848 * (Y - 1900) + 13.1764 * (n_base + i + t / 24));

% 太阳平均经度h
h = deg2rad * (280.19 + 0.2387 * (Y - 1900) + 0.9857 * (n_base + i + t / 24));

% 月球近地点平均经度p1
p1 = deg2rad * (334.39 + 40.6625 * (Y - 1900) + 0.1114 * (n_base + i + t / 24));

% 白道升交点在黄道上经度的负值n_astro
n_astro = deg2rad * (100.84 + 19.3282 * (Y - 1900) + 0.0530 * (n_base + i + t / 24));

% 太阳近地点平均经度p2
p2 = deg2rad * (281.22 + 0.0172 * (Y - 1900) + 0.00005 * (n_base + i + t / 24));

% 平太阴时tau
tau = deg2rad * 15 * t - s + h;

% 输出天文元素
fprintf('平太阴时角τ:%f\n月球平均经度s:%f\n太阳平均经度h:%f\n月球近地点平均经度p1:%f\n白道升交点在黄道上经度的负值n:%f\n太阳近地点平均经度p2:%f\n', tau, s, h, p1, n_astro, p2);

% 基本天文元素矩阵
basic_value = [tau, s, h, p1, n_astro, p2, pi/2]';

% 调整到[0, 2*pi]
basic_value = mod(basic_value, 2*pi); % 使用mod简化代码
















%% 计算各分潮的初始位相
% 每个分潮的初始位相 v0 = sum(u_i * 基本天文元素)
O1_v0 = mod(O1_du * basic_value, 2 * pi);
P1_v0 = mod(P1_du * basic_value, 2 * pi);
K1_v0 = mod(K1_du * basic_value, 2 * pi);
M2_v0 = mod(M2_du * basic_value, 2 * pi);
K2_v0 = mod(K2_du * basic_value, 2 * pi);
Q1_v0 = mod(Q1_du * basic_value, 2 * pi);
S2_v0 = mod(S2_du * basic_value, 2 * pi);
N2_v0 = mod(N2_du * basic_value, 2 * pi);


fprintf('O1 分潮初始位相: %.6f rad\n', O1_v0);
fprintf('P1 分潮初始位相: %.6f rad\n', P1_v0);
fprintf('K1 分潮初始位相: %.6f rad\n', K1_v0);
fprintf('M2 分潮初始位相: %.6f rad\n', M2_v0);
fprintf('K2 分潮初始位相: %.6f rad\n', K2_v0);
fprintf('Q1 分潮初始位相: %.6f rad\n', Q1_v0);
fprintf('S2 分潮初始位相: %.6f rad\n', S2_v0);
fprintf('N2 分潮初始位相: %.6f rad\n', N2_v0);

% 汇总所有分潮的角速率和初始位相
Omega = [omega_O1, omega_P1, omega_K1, omega_M2, omega_K2, omega_Q1, omega_S2, omega_N2];
v0 = [O1_v0, P1_v0, K1_v0, M2_v0, K2_v0, Q1_v0, S2_v0, N2_v0];


% 计算焦点因子和焦点订正角
% K1分潮的焦点因子和焦点订正角（直接计算）
a = [p1, n_astro, p2, pi/2]';
K1_fcosu = 0.0002*cos([-2,-1,0,0]*a) + 0.0001*cos([0,-2,0,0]*a) + ...
            0.0198*cos([0,-1,0,-2]*a) + 1*cos([0,0,0,0]*a) + ...
            0.1356*cos([0,1,0,0]*a) + 0.0029*cos([0,2,0,-2]*a);
K1_fsinu = 0.0002*sin([-2,-1,0,0]*a) + 0.0001*sin([0,-2,0,0]*a) + ...
            0.0198*sin([0,-1,0,-2]*a) + 1*sin([0,0,0,0]*a) + ...
            0.1356*sin([0,1,0,0]*a) + 0.0029*sin([0,2,0,-2]*a);

f_K1 = sqrt(K1_fcosu^2 + K1_fsinu^2);
u_K1 = atan(K1_fsinu / K1_fcosu);

% 打印K1的结果
fprintf('K1分潮的交点因子: %.4f\n', f_K1);
fprintf('K1分潮的交点订正角: %.4f\n', u_K1);

% 其他分潮使用自定义函数计算

% O1分潮
p_O1 = [-0.0058, 0.1885, 1, 0.0002, -0.0064, -0.0010];
delta_u_01 = [0,-2,0,0; 0,-1,0,0; 0,0,0,0; 2,-1,0,0; 2,0,0,0; 2,1,0,0];
[f_O1, u_O1] = f_and_u_calculator(p_O1, delta_u_01);
fprintf('O1分潮的交点因子: %.4f\n', f_O1);
fprintf('O1分潮的交点订正角: %.4f\n', u_O1);

% P1分潮
p_P1 = [0.0008, -0.0112, 1, -0.0015, -0.0003];
delta_u_P1 = [0,-2,0,0; 0,-1,0,0; 0,0,0,0; 2,0,0,0; 2,1,0,0];
[f_P1, u_P1] = f_and_u_calculator(p_P1, delta_u_P1);
fprintf('P1分潮的交点因子: %.4f\n', f_P1);
fprintf('P1分潮的交点订正角: %.4f\n', u_P1);

% M2分潮
p_M2 = [0.0005, -0.0373, 1, 0.0006, 0.0002];
delta_u_M2 = [0,-2,0,0; 0,-1,0,0; 0,0,0,0; 2,0,0,0; 2,1,0,0];
[f_M2, u_M2] = f_and_u_calculator(p_M2, delta_u_M2);
fprintf('M2分潮的交点因子: %.4f\n', f_M2);
fprintf('M2分潮的交点订正角: %.4f\n', u_M2);

% K2分潮
p_K2 = [-0.0128, 1, 0.2980, 0.0324];
delta_u_K2 = [0,-1,0,0; 0,0,0,0; 0,1,0,0; 0,2,0,0];
[f_K2, u_K2] = f_and_u_calculator(p_K2, delta_u_K2);
fprintf('K2分潮的交点因子: %.4f\n', f_K2);
fprintf('K2分潮的交点订正角: %.4f\n', u_K2);

% Q1分潮（与O1相同）
[f_Q1, u_Q1] = f_and_u_calculator(p_O1, delta_u_01);
fprintf('Q1分潮的交点因子: %.4f\n', f_Q1);
fprintf('Q1分潮的交点订正角: %.4f\n', u_Q1);

% S2分潮
f_S2 = 1;
u_S2 = 0;
fprintf('S2分潮的交点因子: %.4f\n', f_S2);
fprintf('S2分潮的交点订正角: %.4f\n', u_S2);

% N2分潮（与M2相同）
[f_N2, u_N2] = f_and_u_calculator(p_M2, delta_u_M2);
fprintf('N2分潮的交点因子: %.4f\n', f_N2);
fprintf('N2分潮的交点订正角: %.4f\n', u_N2);


%% 生成法方程的系数矩阵 A
N = length(y1); % 数据长度
A = zeros(N, 16); % 系数矩阵

for i = 1:8
    A(:, 2*i-1) = cos(Omega(i) * (1:N)' + v0(i)); % 奇数列存储cos项
    A(:, 2*i) = sin(Omega(i) * (1:N)' + v0(i));  % 偶数列存储sin项
end

%% 生成法方程的右侧向量 B
B = y1;

%% 计算法方程的矩阵 F1' 和 F2'
F1_prime = A' * A; % 法方程的左侧矩阵 F1'
F2_prime = A' * B; % 法方程的右侧向量 F2'

%% 求解法方程，得到各分潮的解 x 和模拟潮位 y
X = F1_prime \ F2_prime; % 最小二乘解
y_reported = A * X;      % 模拟潮位

%% 打印结果
fprintf('法方程的系数矩阵 A:\n');
disp(A);  % 显示系数矩阵 A

fprintf('法方程的右侧向量 B:\n');
disp(B);  % 显示右侧向量 B（观测潮位数据）
fprintf('法方程的左侧矩阵 F1'' :\n');  % 使用两个单引号来转义
disp(F1_prime);  % 显示法方程的左侧矩阵 F1'

fprintf('法方程的右侧向量 F2'' :\n');  % 使用两个单引号来转义
disp(F2_prime);  % 显示法方程的右侧向量 F2

fprintf('法方程的解 X:\n');
disp(X);  % 显示法方程的解 X（调和常数）

fprintf('模拟潮位 y:\n');
disp(y_reported);  % 显示模拟潮位 y


%% 提取调和常数
h = sqrt(X(1:2:end).^2 + X(2:2:end).^2); % 调和常数幅值
g = atan2(X(2:2:end), X(1:2:end)); % 调和常数相位

% 分潮名称
tides = {'O1', 'P1', 'K1', 'M2', 'S2', 'N2', 'K2', 'Q1'};

% 输出每个分潮的调和常数
for i = 1:length(tides)
    fprintf('%s 分潮的调和常数幅值 h: %.4f cm, 相位 g: %.4f rad\n', tides{i}, h(i), g(i));
end

%% 绘制自报潮位对比图
figure;
plot(x1, y1, 'b', 'DisplayName', '观测潮位');
hold on;
plot(x1, y_reported, 'r--', 'DisplayName', '自报潮位');
xlabel('时间序列');
ylabel('潮位 (cm)');
title('观测潮位与自报潮位对比');
legend;

%% 潮位预报
N_pred = 24 * 31; % 31天的每小时数据
A_pred = zeros(N_pred, 16);

for i = 1:8
    A_pred(:, 2*i-1) = cos(Omega(i) * (1:N_pred)' + v0(i));
    A_pred(:, 2*i) = sin(Omega(i) * (1:N_pred)' + v0(i));
end

% 计算预报潮位
y_pred = A_pred * X;

%% 绘制预报潮位图
figure;
plot(1:N_pred, y_pred, 'g', 'DisplayName', '预报潮位');
xlabel('时间序列');
ylabel('潮位 (cm)');
title('2003年5月1日至6月1日的潮位预报');
legend;










function [f, u] = f_and_u_calculator(p, delta_u)
    % 计算焦点因子f和焦点订正角u
    % 输入:
    % p - 分潮的引潮力系数比 [p1, n, p2, pi/2]
    % delta_u - 次要分潮与主要分潮的差值矩阵 Δμ
    % 输出:
    % f - 焦点因子
    % u - 焦点订正角
    
    % 计算焦点因子部分
    f_cos = 0;
    f_sin = 0;
    
    for i = 1:size(delta_u, 1)
        % 计算f_cos和f_sin
        f_cos = f_cos + cos(delta_u(i, 1)*p(1) + delta_u(i, 2)*p(2) + delta_u(i, 3)*p(3) + delta_u(i, 4)*p(4));
        f_sin = f_sin + sin(delta_u(i, 1)*p(1) + delta_u(i, 2)*p(2) + delta_u(i, 3)*p(3) + delta_u(i, 4)*p(4));
    end
    
    % 焦点因子的平方和
    f = sqrt(f_cos^2 + f_sin^2);
    
    % 计算焦点订正角
    u = atan(f_sin / f_cos);
end

