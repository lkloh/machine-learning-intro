function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[NUM_SAMPLES NUM_FEATURES] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, NUM_FEATURES);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for cluster_idx=1:K
	indices_for_cluster = idx==cluster_idx;
	num_samples_in_cluster = sum(indices_for_cluster);
	summed_result = X' * indices_for_cluster;
	centroids(cluster_idx, :) = 1.0/num_samples_in_cluster * summed_result;
end



% =============================================================


end

