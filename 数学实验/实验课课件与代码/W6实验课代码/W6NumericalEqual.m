t1 = 0:pi/9:pi;
t2 = linspace(0,pi,10);
t1 == t2
abs(t1-t2) < 1e-12
norm(t1-t2,Inf)
norm(t1-t2)