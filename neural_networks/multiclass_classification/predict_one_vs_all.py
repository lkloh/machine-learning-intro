#!/usr/bin/env python3

import numpy as np

def one_vs_all_predictions(all_theta, X):
    '''
    Predict the label for a trained one-vs-all cllassified.

    The labels are in the range 1, 2, ..., K
    where K = len(all_theta)

    This function returns


%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)
    '''
    (_, num_features) = X.shape
    (num_labels, _) = all_theta.shape

# num_labels = size(all_theta, 1);
#
# % You need to return the following variables correctly
# p = zeros(size(X, 1), 1);
#
# % Add ones to the X data matrix
# X = [ones(m, 1) X];