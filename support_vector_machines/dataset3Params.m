function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

candidate_C_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
candidate_sigma_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

best_prediction_error = 10^7; % large value
for C_idx=1:numel(candidate_C_array)
    for sigma_idx=1:numel(candidate_sigma_array)
        candidate_C = candidate_C_array(C_idx);
        candidate_sigma = candidate_sigma_array(sigma_idx);

        % Train the SVM
        model = svmTrain(X, y, candidate_C, @(x1, x2) gaussianKernel(x1, x2, candidate_sigma));

        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        if (prediction_error < best_prediction_error)
            C = candidate_C;
            sigma = candidate_sigma;
        end
    end
end

% =========================================================================

end
