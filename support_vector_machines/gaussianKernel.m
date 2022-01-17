function sim = gaussianKernel(x1, x2, sigma)
% RBFKERNEL returns a radial basis function kernel between x1 and x2
%    sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%    with bandwidth sigma and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

exponent = sum((x1 - x2).^2) / (2 * sigma^2);
sim = exp(-1 * exponent);
    
end
