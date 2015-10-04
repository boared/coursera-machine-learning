function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
rowSize = size(X, 1);
for row = 1:rowSize
    hx_i = sigmoid( theta.' * X(row, :).' );
    logY1 = log( hx_i );
    logY0 = log( 1 - sigmoid( theta.' * X(row, :).' ) );
    y_i = y(row, 1);
    
    J = J + ( -y_i*logY1 - (1 - y_i)*logY0) + (lambda/(2*m))*( (theta.' * theta) - (theta(1) * theta(1)) );
    
    grad(1, 1) = grad(1, 1) + (hx_i - y_i)*X(row, 1);
    for t = 2:size(theta)
        grad(t, 1) = grad(t, 1) + (hx_i - y_i)*X(row, t) + (lambda/m)*theta(t);
    end
end

J = J / m;
grad = grad ./ m;





% =============================================================

end
