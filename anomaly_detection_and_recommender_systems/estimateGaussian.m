function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[num_samples, num_features] = size(X);

% You should return these values correctly
mu = zeros(num_features, 1);
sigma2 = zeros(num_features, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = mean(X)';


subtracted = X - ones(size(X)) * diag(mu);
sigma2 = (1.0/num_samples) * sum(subtracted.^2)';

fprintf('\nsigma2: %f, var(X): %f \n', sigma2, var(X));


% =============================================================


end
