clear
[A,map]=imread('MyWangJingze.gif','frames','all') ;
b=size(A);
for i = 1:b(4)
    imshow(A(:,:,:,i),map);
    pause(0.05);
%     drawnow;
end