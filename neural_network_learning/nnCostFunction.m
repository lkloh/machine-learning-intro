function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, Y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, Y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
num_samples = size(X, 1);
         
% =================================================================== %
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% =================================================================== %


J = 0;

A1 = add_ones(X);
Z2 = A1 * transpose(Theta1); 
%A2 = arrayfun(@(z) sigmoid(z), Z2);
A2 = sigmoid(Z2);
Z3 = add_ones(A2) * transpose(Theta2); 
A3 = sigmoid(Z3);
H = A3;

for sample_idx = 1:num_samples
    for label_idx = 1:num_labels
        yy = (label_idx == Y(sample_idx));
        hypothesis = H(sample_idx, label_idx);
        J -= yy * log(hypothesis);
        J -= (1 - yy) * log(1 - hypothesis);
    end
end

J *= 1.0 / num_samples;

% =================================================================== %
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1...K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% =================================================================== %


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for sample_idx = 1:num_samples
    % Step 1 - Feedforward pass
    a1 = transpose(X(sample_idx,:));
    theta1_arr = Theta1(:,sample_idx);
    z2 = [1; a1] * transpose(theta1_arr);
    a2 = sigmoid(z2);
    z3 = add_ones(a2) * transpose(Theta2);
    a3 = sigmoid(z3);

    % Step 2 - output layer backpropagation
    delta3 = zeros(num_labels, 1);
    for label_idx = 1:num_labels
        yy = (Y(sample_idx) == sample_idx);
        delta3(label_idx) = a3(label_idx) - yy;
    end

    % Step 3 - hidden layer backpropagation
    delta2 = transpose(Theta2) * delta3 .* sigmoidGradient(z2);

    % Step 4 - accumulate gradients
    Theta2_grad = delta3(2:end) * transpose(a2);
    Theta1_grad = delta2(2:end) * transpose(a1);
end


% =================================================================== %
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% =================================================================== %

J += (lambda / (2.0 * num_samples)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)))

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
