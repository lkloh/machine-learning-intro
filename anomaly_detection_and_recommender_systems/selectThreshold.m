function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = (pval < epsilon);
    %disp(sum(predictions));

    true_positives = sum((predictions == 1)) & (yval == 1); 
    false_positives = sum((predictions == 1)) & (yval == 0); 
    false_negatives = sum((predictions == 0)) & (yval == 1);

    precision = 0;
    if (sum(true_positives) + sum(false_positives)) > 0
      precision = sum(true_positives) / (sum(true_positives) + sum(false_positives));
    end
    
    recall = 0;
    if (sum(true_positives) + sum(false_negatives)) > 0  
      recall = sum(true_positives) / (sum(true_positives) + sum(false_negatives));
    end

    %fprintf('\n true_positive: %i, false positive: %i', sum(true_positives), sum(false_positives));


    if (precision + recall) > 0
      F1 = (2 * precision * recall) / (precision + recall);
    end

    %fprintf("\n precision: %i, recall: %i, f1: %f", precision, recall, F1);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
