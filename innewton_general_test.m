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
%-------ITS HESSIAN, ITS GRADIENT
%-------AND X0: STARTING POINT OF OUR INEXACT NEWTON METHOD
idx = [1:1:n];
f = @(x)sum(x + (x.^4)/4 + (x.^2)/2);
gradf = @(x) (x.^3 + x + 1);
Hessf = @(x) sparse(idx,idx,1+3*(x.^2));
rng(1)
x0 = rand(n,1);

% MAX PCG ITERATIONS
max_pcgiters = 50;

%% RUN THE INEXACT NEWTON (LINEAR) ON f

% OPTIONS FOR FORCING TERMS
fterms = @(gradf, x, k) 0.5;
disp('**** IN_NEWTON: F.T. OPTIONS *****')
disp(['Forcing Terms: ', char(fterms)])

disp('**** IN_NEWTON: START *****')
tic
[xk_lin, fk_lin, gradfk_norm_lin, k_lin, xseq_lin, btseq_lin, norm_grad_seq_lin] = ...
    innewton(x0, f, gradf, Hessf, alpha0, kmax, ...
    tollgrad, c1, rho, btmax, ...
    fterms, max_pcgiters);
toc
disp('**** IN_NEWTON: FINISHED *****')
disp('**** IN_NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_lin), ';'])
disp(['f(xk): ', num2str(fk_lin), ';'])
disp(['N. of Iterations: ', num2str(k_lin),'/',num2str(kmax), ';'])

disp('************************************')



%% RUN THE INEXACT NEWTON (SUPERLINEAR) ON f

% OPTIONS FOR FORCING TERMS
fterms = @(gradf, x, k) min(0.5, sqrt(norm(gradf(x))));

disp('**** IN_NEWTON: F.T. OPTIONS *****')
disp(['Forcing Terms: ', char(fterms)])

disp('**** IN_NEWTON: START *****')
tic
[xk_slin, fk_slin, gradfk_norm_slin, k_slin, xseq_slin, btseq_slin, norm_grad_seq_slin] = ...
    innewton(x0, f, gradf, Hessf, alpha0, kmax, ...
    tollgrad, c1, rho, btmax, ...
    fterms, max_pcgiters);
toc
disp('**** IN.NEWTON: FINISHED *****')
disp('**** IN.NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_slin), ';'])
disp(['f(xk): ', num2str(fk_slin), ';'])
disp(['N. of Iterations: ', num2str(k_slin),'/',num2str(kmax), ';'])
disp('************************************')



%% RUN THE INEXACT NEWTON (QUADRATIC) ON f

% OPTIONS FOR FORCING TERMS
fterms = @(gradf, x, k) min(0.5, norm(gradf(x)));
disp('**** IN.NEWTON: F.T. OPTIONS *****')
disp(['Forcing Terms: ', char(fterms)])

disp('**** IN.NEWTON: START *****')
tic
[xk_q, fk_q, gradfk_norm_q, k_q, xseq_q, btseq_q, norm_grad_seq_q] = ...
    innewton(x0, f, gradf, Hessf, alpha0, kmax, ...
    tollgrad, c1, rho, btmax, ...
    fterms, max_pcgiters);
toc
disp('**** IN.NEWTON: FINISHED *****')
disp('**** IN.NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_q), ';'])
disp(['f(xk): ', num2str(fk_q), ';'])
disp(['N. of Iterations: ', num2str(k_q),'/',num2str(kmax), ';'])
disp('************************************')
