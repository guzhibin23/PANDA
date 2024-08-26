function [X] = logSchatten_Shrink(X, lambda, mode)

sX = size(X);

if mode == 3
    n3 = sX(1);
    m = min(sX(2), sX(3));
else
    n3 = sX(3);
    m = min(sX(1), sX(2));
end


if mode == 1
   Y=X2Yi(X,3);
elseif  mode == 3
   Y = shiftdim(X, 1);
else
   Y = X;
end
Yhat = fft(Y,[],3);

if mod(n3,2) == 0
   endValue = n3/2;
   for i = 1:endValue
       [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
       for j = 1:m
           shat(j,j) = update_log(shat(j,j),lambda); %% update_log(y,beta): solve min_{x>=0} beta*log(1+x)+1/2*(x-y)^2
       end
       Yhat(:,:,i) = uhat*shat*vhat';
       if i > 1 
          Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
       end
    end
    [uhat,shat,vhat] = svd(full(Yhat(:,:,endValue+1)),'econ');
    for j = 1:m
        shat(j,j) = update_log(shat(j,j),lambda);
    end
    Yhat(:,:,endValue+1) = uhat*shat*vhat';
else
    endValue = (n3+1)/2;
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        for j = 1:m
            shat(j,j) = update_log(shat(j,j),lambda);
        end
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1 
           Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
        end
     end
end

    
         
         
           
     
%% Raw IFFT
Y = ifft(Yhat,[],3);
%% New IFFT
% for c=1:n2
%     Y(:,c,:)=ifft(Yhat(:,c,:));
% end
if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end
