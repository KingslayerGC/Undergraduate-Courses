%% 生成一个随机的对称且对角元为0矩阵
n = 1000;
A = rand(n);
A = (A+A')/2;
Diagind = 1:n+1:n^2;
A(Diagind) = 0.0;
%% 假设第一个点为我们的参照点
tic;
IndNotVisited = 2:n;     %记录尚未比较和访问的点
Dist = A(1,:);
for i = 2:n
    [minv,minsp] = min(Dist(IndNotVisited));        %仅在尚未确定路径长度的点进行比较，提高效率
    minp = IndNotVisited(minsp);                    %minsp为子集中的位置，并不是n个点中的位置，需要再调用一次
    Dist(IndNotVisited) = min(Dist(IndNotVisited),minv+A(minp,IndNotVisited));%比较原始距离和用新最短距离更新的和哪个更小，并更新
    IndNotVisited(minsp)=[];                        %将刚刚更新距离的点去掉
end
toc;