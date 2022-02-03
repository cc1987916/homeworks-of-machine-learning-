function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%���������߼��ع�Ĵ��ۺ������ݶ�
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%����J=LRCOSTFUNCTION(theta, X, y, lambda) ������theta��Ϊ�������������߼��ع飬�Լ����ڲ���theta���ݶ�

% Initialize some useful values
%��ʼ��һЩ����
m = length(y); % number of training examples
%m����ѵ������������

% You need to return the following variables correctly 
% ��Ҫ�������µ�������������������ʱ��û�������������������ɾ������
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%�����Ǽ���ָ��theta�µĴ��ۺ������ñ���J������ô��ۺ���
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%�����˼�ǣ��Ƽ�����ʱ�򣬾���ʹ���������ķ���
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%��Ϊû��ʹ���������ʦ����ʾ�ķ�������û��������

%theta = ones(size(X,2)+1,1);
%lambda = 0.01;


%˵�������ȣ�����������������ĸ�����theta��������, X������, y��������,
%lambda����������һ��ʵ����
%theta������ά������ѵ����X��training set��������һ��������ͨ����������n
%������ע���£���Ϊƫ�õ�Ԫ��bias unit���Ĵ��ڣ�����theta��ά����n+1ά��
%X��ѵ��������һ������ά��Ϊm��n��m��ʾѵ����������������n��ʾ��������
%y�Ǹ�������ά��Ϊm��1����ʾX��ÿ��������Ӧ�Ĺ۲�������Ϊ��m��������Ӧ��
%��m��1ά��
%
%������ΪX����ƫ�õ�Ԫ��ͬʱҲ������ҲҪΪtheta����һ��ά�ȡ�
X = [ones(m, 1) X];
theta = [1;theta];
%�������sigmoid����
hypo = sigmoid(X * theta);

reg = lambda/(2*m)*(theta(2:end)'*theta(2:end));

J = (-1/m).*(y'*log(hypo)+(1 .- y)'*log(1.-hypo)) + reg;

grad1 = 1/m.*(X'*(hypo-y))(1);
grad_other = 1/m.*(X'*(hypo-y))(2:end) + lambda/m.*theta(2:end);
grad = [grad1;grad_other];







% =============================================================

grad = grad(:);

end
