function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
num_samples = length(y); % number of training examples
num_features = length(theta);

% ============== Cost of regularized linear regression ============== %

hypothesis = X * theta;
error = (hypothesis - y);

J = 1.0/(2*num_samples) * sum(error.^2) + lambda/(2*num_samples) * sum(theta(2:end).^2);

% ============== Regularized linear regression gradient ============== %

grad = zeros(num_features,1);


grad(1) = (1/num_samples) * dot(error, X(:,1));

grad(2:end) = (1/num_samples) * transpose(error) * X(:,2:end) + (lambda/num_samples) * theta(2:end);

% =========================================================================

grad = grad(:);

end
