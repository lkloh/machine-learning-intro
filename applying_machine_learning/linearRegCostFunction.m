function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
num_samples = length(y); % number of training examples
num_features = length(theta) - 1;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_error_1 = 0;
for sample_idx = 1:num_samples
    xx = X(sample_idx,:);
    yy = y(sample_idx);

    h = dot(theta, xx);
    J_error_1 += (h - yy)^2;
end

J_error_2 = sum(theta(2:end).^2);

J = 1.0/(2*num_samples) * J_error_1 + lambda/(2*num_samples) * J_error_2;














% =========================================================================

grad = grad(:);

end
