function K = RBF_Kernel(X,kernelFunction)
    m = size(X, 1);
    K = zeros(m, m);
    for i=1:m
      for j=1:m
        temp = sum((X(i,:)- X(j,:)).^2, 2);
        K(i, j) = temp;
      end;
    end;
    K = kernelFunction(1, 0) .^ K;
  
  
end;


%call Func : RBF_Kernel(X,@(x1, x2) gaussianKernel(x1, x2, 0.1))