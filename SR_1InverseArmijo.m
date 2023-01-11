% Author: (Sinan Çavuşoğlu)
%
% Description:
% This is a function that implements Symmetric Rank-1 (SR1) update to compute 
% the inverse Hessian approximation at each iteration. Use Armijo Condition 
% to find step length
% 
% Input:
% fhandle: Derivative of Rosenbrock banana function.
% x0: A column vector containing the initial guess for the minimizer.
% tol: Tolerance for the gradient of the objective function.
% H0: A positive definite matrix that represents the initial approximation 
% of the Hessian of the objective function.
% maxit: The maximum number of iterations allowed. 
% alpha0: Initial step size for the Armijo line search.
% mu: The step size shrinkage factor for the Armijo line search.
% amax: The maximum number of iterations allowed in the Armijo line search.
%
% Output:
% X: A matrix containing the iterates of the minimizer.,
% Grad: A vector containing the norm of the gradient at each iteration.
% ite: The number of iterations performed.

clear all
clc

f = @(x1,x2) 100*(x2 - x1^2)^2 + (1-x1)^2;
fhandle = @(x1,x2) [-400*x1*(x2-x1^2)-(2*(1 - x1)) ; 200*(x2-x1^2)];
% 3 different tolerance
tol1 = 10^-3;
tol2 = 10^-6;
tol3 = 10^-9;
% 2 different initial guess
x0_1 = transpose([-0.5, 1]);
x0_2 = transpose([1.1, 1.1]);
%%%
H0 = eye(2);
maxit = 10000;
alpha0 = 1;
c = 10^-4;
mu = 0.5;
amax = 100;
%%%


% For x0_1
% Tolerance 1
[X1,Grad1,ite1] = SR1_inverse(fhandle,x0_1,tol1,H0,maxit,alpha0,c,mu,amax);
%Tolerance 2
[X2,Grad2,ite2] = SR1_inverse(fhandle,x0_1,tol2,H0,maxit,alpha0,c,mu,amax);
%Tolerance 3
[X3,Grad3,ite3] = SR1_inverse(fhandle,x0_1,tol3,H0,maxit,alpha0,c,mu,amax);

% For X0_2
% Tolerance 1
[X4,Grad4,ite4] = SR1_inverse(fhandle,x0_2,tol1,H0,maxit,alpha0,c,mu,amax);
% Tolerance 2
[X5,Grad5,ite5] = SR1_inverse(fhandle,x0_2,tol2,H0,maxit,alpha0,c,mu,amax);
% Tolerance 3
[X6,Grad6,ite6] = SR1_inverse(fhandle,x0_2,tol3,H0,maxit,alpha0,c,mu,amax);

% Table
fprintf('For the initial value (-0.5,1)\n')
fprintf('  Tol | iteration  |      x value     |  Norm_Gradient\n');
fprintf('----- |------------|------------------|---------------\n');
fprintf(' 1e-3 |    %3i     |    %1.12f   %1.12f     |     %1.12f \n',ite1,X1(:,end)',Grad1(end));
fprintf(' 1e-6 |    %3i     |    %1.12f   %1.12f     |     %1.12f \n',ite2,X2(:,end)',Grad2(end));
fprintf(' 1e-9 |    %3i     |    %1.12f   %1.12f     |     %1.12f \n',ite3,X3(:,end)',Grad3(end));

fprintf('For the initial value (1.1,1.1)\n')
fprintf('  Tol | iteration  |               x value                  |  Norm_Gradient\n');
fprintf('----- |------------|----------------------------------------|-----------------\n');
fprintf(' 1e-3 |    %3i     |    %1.12f   %1.12f     |  %1.12f \n',ite4,X4(:,end)',Grad4(end));
fprintf(' 1e-6 |    %3i     |    %1.12f   %1.12f     |  %1.12f \n',ite5,X5(:,end)',Grad5(end));
fprintf(' 1e-9 |    %3i     |    %1.12f   %1.12f     |  %1.12f \n',ite6,X6(:,end)',Grad6(end));

function [X,Grad,ite] = SR1_inverse(fhandle,x0,tol,H0,maxit,alpha0,c,mu,amax)
f = @(x1,x2) 100*(x2 - x1^2)^2 + (1-x1)^2;
grad = fhandle(x0(1), x0(2));
ite = 0;
while (ite <= maxit) && (norm(grad) >= tol)
    %Search Direction
    p = -H0 * grad;

    %Armijo Condition
    f_x0 = f(x0(1), x0(2));
    var_arm = x0 + alpha0 * p;
    f_armijo_new = f(var_arm(1), var_arm(2));
    j = 0;
    while (f_armijo_new >= f_x0 + c*alpha0*transpose(grad)*p) && j<=amax
        alpha0 = alpha0 * mu;
        j = j + 1;
        var_arm = (x0 + alpha0 * p);
        f_armijo_new = f(var_arm(1), var_arm(2));
    end

    % Update the Soln
    x_new = x0 + alpha0*p;
    s = x_new - x0;
    y = fhandle(x_new(1), x_new(2)) - fhandle(x0(1), x0(2));

    H0 = H0 + (((s - H0*y)*transpose(s - H0*y))/(transpose(s - H0*y)*y));

    % Update Variable
    x0 = x_new;
    grad = fhandle(x0(1), x0(2));
    Grad(ite+1) = norm(fhandle(x0(1), x0(2)));
    ite = ite + 1;
    alpha0 = 1;
    X(1:2,ite) = x0;
end
end
