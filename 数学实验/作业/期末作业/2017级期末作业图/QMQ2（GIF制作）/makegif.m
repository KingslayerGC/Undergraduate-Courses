clear;clc;
figure %新建一张图
axis([0 5 0 2])%定义x轴（从0到5）和y轴的范围（从0到2）
for i=1:4
    if i==1
        text(i,1,'数','fontsize',40,'color','red');%i=1时，写一个‘数’字
    end
    if i==2
        text(i,1,'学','fontsize',40,'color','red');%i=2时，写一个‘学’字
    end
    if i==3
        text(i,1,'实','fontsize',40,'color','red'); %i=3时，写一个‘实’字
    end
    if i==4
        text(i,1,'验','fontsize',40,'color','red');%i=4时，写一个‘验’字
    end    
    picname=[num2str(i) '.fig'];%保存的文件名：如i=1时，picname=1.fig
    hold on % 写后面的字时，不把前面的字冲掉
    saveas(gcf,picname)
end


stepall=4;
for i=1:stepall
    picname=[num2str(i) '.fig'];
    open(picname)
%     set(gcf,'outerposition',get(0,'screensize'));% matlab窗口最大化
    frame=getframe(gcf);  
    im=frame2im(frame);%制作gif文件，图像必须是index索引图像  
    [I,map]=rgb2ind(im,20);          
    if i==1
        imwrite(I,map,'shuxueshiyan.gif','gif', 'Loopcount',inf,'DelayTime',0.5);%第一次必须创建！
    elseif i==stepall
        imwrite(I,map,'shuxueshiyan.gif','gif','WriteMode','append','DelayTime',0.5);
    else
        imwrite(I,map,'shuxueshiyan.gif','gif','WriteMode','append','DelayTime',0.5);
    end;  
    close all
end