function y = rank_fun_derivative(x,a)
% y = (1+a)*a./(a+x).^2;
temp=a*exp(x/a).*((x+exp(-x/a)).^2);%exp(1)是自然对数e
y=(x+a)./temp;
end
