function [lambda_vec, error_train, error_validate] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda
lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 5, 10];

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_validate = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)

for lambda_idx = 1:length(lambda_vec)
    lambda = lambda_vec(lambda_idx);

    [theta] = trainLinearReg(X, y, lambda);

    [J_train, _] = linearRegCostFunction(X, y, theta, lambda);
    [J_validate, _] = linearRegCostFunction(Xval, yval, theta, lambda);

    error_train(lambda_idx) = J_train;
    error_validate(lambda_idx) = J_validate;
end


% =========================================================================

end
