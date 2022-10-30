function [xk, normFk, k, xseq, btseq] = ...
    newtonsolve_bcktrck(x0, F, JF, ...
    kmax, Ftol, c1, rho, btmax)
%
% function [xk, normFk, k, xseq, btseq] = ...
%     newtonsolve_bcktrck(x0, F, JF, Hessf, alpha0, ...
%     kmax, tollgrad, c1, rho, btmax)
%
% Function that performs the newton method, 
% implementing the backtracking strategy, for solving a 
% nonlinear system F(x)=0.
% Backtracking strategy is implemented w.r.t. the merit function 
% f(x) = 0.5*norm(F(x))^2
%
% INPUTS:
% x0 = n-dimensional column vector;
% F = function handle that describes a function R^n->R^n;
% JF = function handle that describes the Jacobian of F;
% kmax = maximum number of iterations permitted;
% Ftol = value used as stopping criterion w.r.t. the norm of the
% value of F in xk;
% c1 = the factor of the Armijo condition that must be a scalar in (0,1);
% rho = fixed factor, lesser than 1, used for reducing alpha0;
% btmax = maximum number of steps for updating alpha during the 
% backtracking strategy.
%
% OUTPUTS:
% xk = the last x computed by the function;
% normFk = the value norm(F(xk));
% gradfk_norm = value of the norm of gradf(xk)
% k = index of the last iteration performed
% xseq = n-by-k matrix where the columns are the xk computed during the 
% iterations
% btseq = 1-by-k vector where elements are the number of backtracking
% iterations at each optimization step.
%

alpha0 = 1;

f = @(x) 0.5 * norm(F(x))^2;
gradf = @(x) JF(x)'*F(x);

% Function handle for the armijo condition
farmijo = @(fk, alpha, xk, pk) ...
    fk + c1 * alpha * gradf(xk)' * pk;

% Initializations
xseq = zeros(length(x0), kmax);
btseq = zeros(1, kmax);

xk = x0;
k = 0;
normFk = norm(F(xk));

while k < kmax && normFk >= Ftol
    % Compute the Newton step 
    % (descent direction of f)
    % solving the linear system
    % JF(xk) delta_xk = - F(xk)
    delta_xk = -JF(xk)\F(xk);
    
    % Reset the value of alpha
    alpha = alpha0;
    
    % Compute the candidate new xk
    xnew = xk + alpha * delta_xk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy: 
    % 2nd condition is the Armijo condition not satisfied
    while bt < btmax && fnew > farmijo(f(xk), alpha, xk, delta_xk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew w.r.t. the reduced alpha
        xnew = xk + alpha * delta_xk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
        
    end
    
    % Update xk, normFk
    xk = xnew;
    normFk = norm(F(xk));
    
    % Increase the step by one
    k = k + 1;
    
    % Store current xk in xseq
    xseq(:, k) = xk;
    % Store bt iterations in btseq
    btseq(k) = bt;
end

% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
btseq = btseq(1:k);

end