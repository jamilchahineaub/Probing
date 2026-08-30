function upper = binomial_upper95(k,n)
% Exact one-sided 95% Clopper-Pearson upper bound without toolboxes.
if n<=0 || k>=n, upper=1; return; end
alpha=.05; lo=k/n; hi=1;
for q=1:80
    mid=(lo+hi)/2;
    if binocdf_base(k,n,mid)>alpha, lo=mid; else, hi=mid; end
end
upper=hi;
end
function value=binocdf_base(k,n,p)
if p<=0, value=1; return; elseif p>=1, value=double(k>=n); return; end
j=(0:k)'; logs=gammaln(n+1)-gammaln(j+1)-gammaln(n-j+1)+j*log(p)+(n-j)*log1p(-p); a=max(logs); value=min(1,exp(a)*sum(exp(logs-a)));
end
