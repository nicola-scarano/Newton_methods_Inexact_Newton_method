%% LOADING THE VARIABLES FOR THE TEST

clear 
clc

alpha0 = 1;
c1 = 1e-4;
rho = 0.8;
btmax = 50;
tollgrad = 10e-13;
kmax = 100;
n = 1e4;

%-------DEFINITION OF THE FUNCTIONAL TO BE ANALYZED,
%-------ITS HESSIAN, ITS GRADIENT;
%-------AND X0: STARTING POINT OF OUR NEWTON METHOD
idx = [1:1:n];
f = @(x)sum(x + (x.^4)/4 + (x.^2)/2);
gradf = @(x) (x.^3 + x + 1);
Hessf = @(x) sparse(idx,idx,1+3*(x.^2));
rng(1)
x0 = rand(n,1);


%% RUN THE NEWTON_DIRECT ON f
tic
disp('**** NEWTON_DIRECT: START *****')
[xk_n, fk_n, gradfk_norm_n, k_n, xseq_n, btseq_n] = ...
    newton_direct(x0, f, gradf, Hessf, alpha0, kmax, ...
    tollgrad, c1, rho, btmax);
toc
disp('**** NEWTON_DIRECT: FINISHED *****')
disp('**** NEWTON_DIRECT: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n), ' (actual minimum: [0; 0]);'])
disp(['f(xk): ', num2str(fk_n), ' (actual min. value: 5);'])
disp(['N. of Iterations: ', num2str(k_n),'/',num2str(kmax), ';'])
disp('************************************')


%% RUN THE NEWTON_PCG ON f

disp('**** NEWTON_PCG: START *****')
tic
[xk_ng, fk_ng, gradfk_norm_ng, k_ng, xseq_ng, btseq_ng, norm_grad_seq_ng] = ...
    newton_pcg(x0, f, gradf, Hessf, alpha0, kmax, ...
    tollgrad, c1, rho, btmax);

toc
disp('**** NEWTON_PCG: FINISHED *****')
disp('**** NEWTON_PCG: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_ng), ' (actual minimum: [0; 0]);'])
disp(['f(xk): ', num2str(fk_ng), ' (actual min. value: 5);'])
disp(['N. of Iterations: ', num2str(k_ng),'/',num2str(kmax), ';'])
disp('************************************')



