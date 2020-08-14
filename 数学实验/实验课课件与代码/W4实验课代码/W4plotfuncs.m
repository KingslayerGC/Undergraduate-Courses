%% W3作业Q3，使用ezplot可能会出现断线的错误，这是程序设计的不足
%% 符号函数绘图法
close all, clear all
syms x y z
ezplot('sin(x)*exp(x)'); %自动调节坐标轴范围
figure;
fplot(sin(x)*exp(x));   %坐标轴范围固定在[-5,5],无title

%% 匿名函数绘图法
clear all
figure;
f = @(x) sin(x).*exp(x);
ezplot(f);
figure;
fplot(f);

