function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%计算正则化逻辑回归的代价函数及梯度
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%函数J=LRCOSTFUNCTION(theta, X, y, lambda) ，能以theta作为参数计算正则化逻辑回归，以及关于参数theta的梯度

% Initialize some useful values
%初始化一些参数
m = length(y); % number of training examples
%m代表训练样本的数量

% You need to return the following variables correctly 
% 需要返还如下的两个变量，（我做的时候没有用下面的两个，可以删除掉）
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%本题是计算指定theta下的代价函数，用变量J来保存该代价函数
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%大概意思是，推荐计算时候，尽量使用向量化的方法
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%因为没有使用吴恩达老师的提示的方法，就没不翻译了

%theta = ones(size(X,2)+1,1);
%lambda = 0.01;


%说明，首先，这个函数的输入是四个变量theta（向量）, X（矩阵）, y（向量）,
%lambda（标量或者一个实数）
%theta向量的维度是由训练集X（training set）决定的一个向量，通常由其列数n
%来决定注意下，因为偏置单元（bias unit）的存在，所以theta的维度是n+1维。
%X是训练集，是一个矩阵，维度为m×n，m表示训练集的样本个数，n表示特征数。
%y是个向量，维度为m×1，表示X中每个样本对应的观测结果，因为有m个样本对应就
%是m×1维。
%
%下面是为X增加偏置单元，同时也别忘了也要为theta增加一个维度。
X = [ones(m, 1) X];
theta = [1;theta];
%下面计算sigmoid函数
hypo = sigmoid(X * theta);

reg = lambda/(2*m)*(theta(2:end)'*theta(2:end));

J = (-1/m).*(y'*log(hypo)+(1 .- y)'*log(1.-hypo)) + reg;

grad1 = 1/m.*(X'*(hypo-y))(1);
grad_other = 1/m.*(X'*(hypo-y))(2:end) + lambda/m.*theta(2:end);
grad = [grad1;grad_other];







% =============================================================

grad = grad(:);

end
