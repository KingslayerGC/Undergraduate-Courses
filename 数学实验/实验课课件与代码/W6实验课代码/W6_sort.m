A=[5,1,7;2,8,6;9,3,4]
AS = sort(A)                %按列升序排序
ASD = sort(A,'descend')     %按列降序排序
AS2 = sort(A,2)             %按行升序排序

ASF = reshape(sort(A(:)),3,3) %全部元素排序
