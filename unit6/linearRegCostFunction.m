function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypo = X*theta;
Matr_Err = 1/2.*(hypo - y).^2;
reg = 1/2.*theta(2).^2
J = (1/m)*(sum(sum(Matr_Err)) + lambda*sum(reg));

grad = X'*(hypo - y);
grad_reg = grad(2) + lambda.*theta(2);
grad = (1/m)*[grad(1) grad_reg];








% =========================================================================

grad = grad(:);

end
