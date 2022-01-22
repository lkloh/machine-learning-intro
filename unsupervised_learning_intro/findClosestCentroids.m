function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

NUM_SAMPLES = size(X,1);

% You need to return the following variables correctly.
idx = zeros(NUM_SAMPLES, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for sample_idx = 1:NUM_SAMPLES
	sample_vec = X(sample_idx,:);

	min_cost = intmax;
	min_cost_cluster = 1;
	for cluster_idx = 1:K
		centroid_vec = centroids(cluster_idx,:);

		current_cost = sum((sample_vec - centroid_vec).^2);
		if current_cost < min_cost
			min_cost = current_cost;
			min_cost_cluster = cluster_idx;
		end
	end

	idx(sample_idx) = min_cost_cluster;
end





% =============================================================

end

