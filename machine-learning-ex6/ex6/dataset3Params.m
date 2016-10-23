function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;



C =  0.27000;
sigma =  0.090000;
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



% =========================================================================

end
%c = C;
%sig = sigma;
%error = 10000;
%while (C <= 40)
%  sigma = 0.01;
%  while(sigma <= 40)
%    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%    visualizeBoundary(X, y, model);
%    prediction = svmPredict(model, Xval);
%    errorCurrent = mean(prediction ~= yval);
%    disp(errorCurrent);
%    disp(C);
%    disp(sigma);
%    if (error > errorCurrent)
%      error = errorCurrent;
%      c = C;
%      sig = sigma;
%    endif
%    disp(error);
%    sigma = sigma * 3;
%  endwhile
%  C = C * 3;
%endwhile
%C = c;
%sigma = sig;
%disp('-----');
%disp(C);
%disp(sigma);