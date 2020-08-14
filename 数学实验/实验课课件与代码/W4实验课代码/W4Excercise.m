clear all, close all
%% 问题1
syms x
r = taylor(atan(x),x,0,'Order',11)

%% 问题2
syms x y(x)
S1 = dsolve(diff(y(x))==y+x)
S2 = dsolve(diff(y(x))==y^2+x^2)

%% 问题3
R1 = vpa(sym(pi),100)
R2 = vpa(sym(pi),101)
R3 = vpa(sym(pi),102)
R4 = vpa(0.1,10)