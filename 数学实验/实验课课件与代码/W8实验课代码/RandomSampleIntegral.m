%% 生成随机矩阵
rng default
x = rand(1,1000);
x = [x,0,1];
y = x.^2 + x;
%% 对数据根据x的大小进行排序
[xs ind] = sort(x);
ys = y(ind);
GT = 5/6;
%% 使用trapz完成的梯形公式的方法
S1 = trapz(xs,ys);
err1 = abs(S1 - GT)
%% 自主实现的梯形公式方法(左矩形右矩形请同学们自己思考完成）
l = xs(2:end) - xs(1:end-1);
f = (ys(2:end) + ys(1:end-1))/2;
S2 = sum(l.*f);
err2 = abs(S2 - GT)
