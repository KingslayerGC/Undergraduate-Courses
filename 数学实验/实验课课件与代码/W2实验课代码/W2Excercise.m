clear all
%% 问题1
a = 1/0
b = -1/0
c = 0/0

%% 问题2
p = [1 0 4.75 2.5];
R = roots(p)
real_root = R(imag(R)==0)
nthroot(p,3);%这一句并不对，只是求了所有系数的三次方根
syms x
f = x^3 + 4.75*x + 2.5;
S1 = solve(f)
syms x real
f = x^3 + 4.75*x + 2.5;
S2 = solve(f)
% 另一种方法
% fun = @(x) x^3 + 4.75 * x + 2.5; %定义了函数句柄
% real_root_2 = fzero(fun,0) %以0为起点求解一个附近的根

%% 问题3
clear all %不清空Mupad内存，x仍限定为实数
syms n x
f = symsum((n+1)/factorial(n)*x^n,n,1,inf)
