function XX = add_ones(X)
    num_rows = size(X, 1);
    XX = [ones(num_rows, 1) X];
end
