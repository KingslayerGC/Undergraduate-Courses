clear all
format
a = sqrt(2)
format long e
a
b = sqrt(sym(2))
c = sym(sqrt(2))
abs(a-b)    %新版的MATLAB会自动匹配sqrt(2)的真实值，所以居然误差是0！
abs(a-c)
format

format
a = sqrt(979797)
format long e
a
b = sqrt(sym(979797))
c = sym(sqrt(979797))
vpa(abs(a-b))    %不相等了！
vpa(abs(a-c))
format