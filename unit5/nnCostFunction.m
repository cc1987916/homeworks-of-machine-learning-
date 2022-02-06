function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%˵���£�input_layer_size����㵥Ԫ����hidden_layer_size��ʾ�����㵥Ԫ����
%num_labels��ʾ����㵥Ԫ����nn_params������Ĳ���������������һ��չ����������
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
%�������δ����ǽ�չ��������������ɲ�������theta1��theta2������ע���£����迼��bias unit
%��Ϊ�������л��Զ�����bias unit��
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
          hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m,1) X];
net_h = X*Theta1';
out_h = sigmoid(net_h);

out_h = [ones(m,1) out_h];%ά��Ϊ5000*26
net_o = out_h*Theta2';
out_o = sigmoid(net_o);

%˵����out_0��һ��5000*10ά�ȵľ���y��5000*1��Ϊʲô��sum��ͣ�ԭ����ǰ������ó����Ǹ�������
J = -1/m*sum((y'*log(out_o)) + (1-y)'*log(1-out_o));




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
%
err_term_o = zeros(m,num_labels)%ά��Ϊ5000*10
for i =1:num_labels,
err_term_o(:,i) = out_o(:,i) - (y==i);
end;
%err_term_o = err_term_o .*out_o.*(1-out_o);
err_term_h = (err_term_o*Theta2).*out_h.*(1-out_h);%ά��Ϊ5000*26

Theta2_grad = err_term_o'*out_h;%out_hά��Ϊ5000*26  Theta2_gradά��Ϊ10*26

%ǰ�򴫲�ʱ�����bias unit�����򴫲�ʱ��ȥ����һ��
Theta1_grad = err_term_h(:,2:end)'*X;%Theta1_gradά��Ϊ25*401

disp(Theta2_grad);

disp(Theta1_grad);
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta2_reg = Theta2_grad(:,2:end) + lambda.*Theta2(:,2:end);
Theta2_grad = 1/m.*[Theta2_grad(:,1) Theta2_reg];


Theta1_reg = Theta1_grad(:,2:end) + lambda.*Theta1(:,2:end);
Theta1_grad = 1/m.*[Theta1_grad(:,1) Theta1_reg];





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
