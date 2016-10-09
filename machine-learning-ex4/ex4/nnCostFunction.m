function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
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
m = size(X, 1);
X = [ones(m, 1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
t1D = zeros(size(Theta1));
t2D = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

V1 = realpow(1+ exp(-1*X*transpose(Theta1)), -1);
V2 = [ones(m, 1) V1];
V = realpow(1+ exp(-1*V2*transpose(Theta2)), -1);
C = zeros(m,1);
K = size(V)(2);
for i = 1 : m
  yy = zeros(K,1);
  yy(y(i)) = 1;
  C(i) = -1*reallog(V(i,:))*yy - reallog(1 - V(i,:))*(1 - yy);
  dout = transpose(V(i,:)) - yy;
  %dhid = transpose(Theta2(:,2:size(Theta2)(2)))*dout.*transpose(sigmoidGradient(V1(i,:)));
  %size(transpose(Theta2)*dout)
  %size(transpose(sigmoidGradient(V2(i,:))))
  %size(transpose(Theta2)*dout)
  %size(sigmoidGradient(X(i,:)*transpose(Theta1)))
  dhid = (transpose(Theta2)*dout).*([1; transpose(sigmoidGradient(X(i,:)*transpose(Theta1)))]);
  dhid = dhid(2:size(dhid)(1),:);
  size(dhid)
  %disp('---');
  t1D = t1D + dhid*X(i,:);
  t2D = t2D + dout*V2(i,:);
endfor
%disp('-------');
%disp(C);
%disp('--------');
%disp(size(Theta1));
%disp(size(Theta2));
J = sum(C)/m + (sum(sum(Theta1(:,2:size(Theta1)(2)).*Theta1(:,2:size(Theta1)(2)))) + sum(sum(Theta2(:,2:size(Theta2)(2)).*Theta2(:,2:size(Theta2)(2)))))*lambda/(2*m);
tt1 = Theta1;
tt2 = Theta2;
tt1(:,1) = 0;
tt2(:,1) = 0;
Theta1_grad = t1D/m + (tt1)*lambda/m;
Theta2_grad = t2D/m + (tt2)*lambda/m;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
