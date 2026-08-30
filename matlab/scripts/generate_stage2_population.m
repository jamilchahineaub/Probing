function cases = generate_stage2_population(seed_range,n,bounds,prefix)
% Log-Latin-hypercube population using MATLAB's independent RNG stream.
seeds=seed_range(1):seed_range(2); total=numel(seeds)*n;
id=strings(total,1); seed=zeros(total,1); index=zeros(total,1); k=zeros(total,1); c=k; m=k; q=0;
limits=[bounds.k; bounds.c; bounds.m];
for s=seeds
    rng(s,'twister'); coords=zeros(n,3);
    for j=1:3
        bins=((0:n-1)'+rand(n,1))/n; coords(:,j)=bins(randperm(n));
    end
    vals=exp(log(limits(:,1))'+coords.*(log(limits(:,2))-log(limits(:,1)))');
    for i=1:n
        q=q+1; id(q)=sprintf('%s_s%d_c%02d',prefix,s,i-1); seed(q)=s; index(q)=i-1;
        k(q)=vals(i,1); c(q)=vals(i,2); m(q)=vals(i,3);
    end
end
cases=table(id,seed,index,k,c,m,'VariableNames',{'target_id','seed','case_index','k','c','m'});
end
