n = 1000;
A = rand(n);
AA = A;
tic;ind = find(A>=0.5);toc;
tic;B = A>=0.5;toc;
tic;A(ind) = -A(ind);toc;
A = AA; tic;A(find(A>=0.5)) = - A(find(A>=0.5));toc;
A = AA;tic;A(A>=0.5) = - A(A>=0.5);toc;
A = AA;tic;A(B) = - A(B);toc;

n = 1000;
A = rand(n);
AA = A;
tic;ind = find(A>=0.99);toc;
tic;B = A>=0.99;toc;
tic;A(ind) = -A(ind);toc;
A = AA;tic;A(find(A>=0.99)) = - A(find(A>=0.99));toc;
A = AA;tic;A(A>=0.99) = - A(A>=0.99);toc;
A = AA;tic;A(B) = - A(B);toc;