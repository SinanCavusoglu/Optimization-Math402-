% Author: (Sinan Çavuşoğlu)
%
% Description:
% This is a function that uses the conjugate gradient method to solve 
% a linear system of equations Ax=b.
% 
% Input:
% A: Matrix which is coefficients of the system of linear equations.
% b: Vector which is the right hand side of the system of linear equations.
% x0: An initial guess for the solution of the system of linear equations.
% tol: a tolerance value for the convergence of the solution.
% maxit: the maximum number of iterations 
% 
% Output:
% X: a matrix containing the solution of the linear system at each iteration.
% res: a vector containing the residual norm at each iteration.
% ite: the number of iterations required for the solution to converge.

clear all
clc
for n = [5,8,12,20]
A = hilb(n);
b = ones(1,n)';
x0 = zeros(1,n)';
tol = 10^-6;
maxit = 1000;

if n == 5
[X5,res5,ite5] = conj_grad(A,b,x0,tol,maxit);
end
if n == 8
[X8,res8,ite8] = conj_grad(A,b,x0,tol,maxit);
end
if n == 12
[X12,res12,ite12] = conj_grad(A,b,x0,tol,maxit);
end
if n == 20
[X20,res20,ite20] = conj_grad(A,b,x0,tol,maxit);
end
end

fprintf('For the initial value (-1,1)\n')
fprintf(' n | ite |                 x value                |  Norm_of_residual\n');
fprintf('---|-----|----------------------------------------|------------------\n');
fprintf(' 5 | %3i |%1.3f %1.3f %1.3f %1.3f %1.3f| %1.12f \n',ite5,X5(:,end)',res5(end));
fprintf('\n')
fprintf(' n | ite |                                x value                                |  Norm_of_residual\n');
fprintf('---|-----|-----------------------------------------------------------------------|------------------\n');
fprintf(' 8 | %3i |%1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f| %1.12f \n',ite8,X8(:,end)',res8(end));


function [X,res,ite] = conj_grad(A,b,x0,tol,maxit)
ite = 0;
r0 = A*x0 -b;
p0 = -r0;
while (norm(r0) >= tol) && (ite<= maxit)
%Method 
alpha = (transpose(r0)*r0)/(transpose(p0)*A*p0);
x_new = x0 + alpha*p0;
r_new = r0 + alpha*A*p0;
beta = (transpose(r_new) * r_new)/(transpose(r0)*r0);
p_new = -r_new + beta*p0;

%Update 
x0 = x_new;
r0 = r_new;
p0 = p_new;
ite = ite + 1;

X(1:size(x0,1),ite) = x0;
res(ite) = norm(r0);

end
end