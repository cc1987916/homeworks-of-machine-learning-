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
vec_const = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C_optimal = inf;
sigma_optimal = inf;
err_min = inf;

for i = 1:length(vec_const),
    C_curr = vec_const(i);
    for j = 1:length(vec_const),
      sigma_curr = vec_const(j);
      model = svmTrain(X, y, C_curr, @(x1, x2) gaussianKernel(x1, x2, sigma_curr));
      predictions = svmPredict(model, Xval);
      pre_err = mean(double(predictions ~= yval));
      if pre_err < err_min
        C_optimal = C_curr;
        sigma_optimal = sigma_curr;
        err_min = pre_err;
      endif;
    end;
end;






% =========================================================================

end
