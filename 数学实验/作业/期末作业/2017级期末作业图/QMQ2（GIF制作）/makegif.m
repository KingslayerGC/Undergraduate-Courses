clear;clc;
figure %�½�һ��ͼ
axis([0 5 0 2])%����x�ᣨ��0��5����y��ķ�Χ����0��2��
for i=1:4
    if i==1
        text(i,1,'��','fontsize',40,'color','red');%i=1ʱ��дһ����������
    end
    if i==2
        text(i,1,'ѧ','fontsize',40,'color','red');%i=2ʱ��дһ����ѧ����
    end
    if i==3
        text(i,1,'ʵ','fontsize',40,'color','red'); %i=3ʱ��дһ����ʵ����
    end
    if i==4
        text(i,1,'��','fontsize',40,'color','red');%i=4ʱ��дһ�����顯��
    end    
    picname=[num2str(i) '.fig'];%������ļ�������i=1ʱ��picname=1.fig
    hold on % д�������ʱ������ǰ����ֳ��
    saveas(gcf,picname)
end


stepall=4;
for i=1:stepall
    picname=[num2str(i) '.fig'];
    open(picname)
%     set(gcf,'outerposition',get(0,'screensize'));% matlab�������
    frame=getframe(gcf);  
    im=frame2im(frame);%����gif�ļ���ͼ�������index����ͼ��  
    [I,map]=rgb2ind(im,20);          
    if i==1
        imwrite(I,map,'shuxueshiyan.gif','gif', 'Loopcount',inf,'DelayTime',0.5);%��һ�α��봴����
    elseif i==stepall
        imwrite(I,map,'shuxueshiyan.gif','gif','WriteMode','append','DelayTime',0.5);
    else
        imwrite(I,map,'shuxueshiyan.gif','gif','WriteMode','append','DelayTime',0.5);
    end;  
    close all
end