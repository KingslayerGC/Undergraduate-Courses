%% 生成鸭子的坐标
clear;
rng default;
num_d = 4;       %鸭子的数量
num_sample = 1e6;%投点的数量
% r = rand(num_sample,num_d); %其实半径并非均匀分布，但对于结果并没有影响,可忽略之
theta = rand(num_sample,num_d)*2*pi; %关键在于对角度的模拟，每行4列代表四只鸭子的角度
%% 计算相邻鸭子的相差角度
theta_s = sort(theta,2);%对每一行按从小到大的次序排序
theta_diff = theta_s(:,2:num_d)-theta_s(:,1:num_d-1);
%第二列减第一列，第三列减第二列，第四列减第三列
theta_diff = [theta_diff,theta_s(:,1)+2*pi-theta_s(:,num_d)];
%将第一列+2*pi减第四列也加上去
%% 所有鸭子落在一个半圆内，当且仅当至少有一个相邻角度大于等于pi
criteria = theta_diff>=pi;    %大于等于pi的角度会设为1，否则为0
criteria_s = sum(criteria,2); %每行求和，如果符合条件则有一个1，否则为0
Na = sum(criteria_s>=1);       %符合条件的鸭子计数
prob = Na/num_sample
gt = num_d*(1/2)^(num_d-1);
err = abs(prob-gt)