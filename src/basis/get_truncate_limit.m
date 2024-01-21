function [L,S] = get_truncate_limit(L, n, c, ratio)
S_max = 200;
S = zeros(L+1,1);
sphbes_zeros = zeros([L+1,S_max]);

for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,S_max,1);
    S(l+1) = find(sphbes_zeros(l+1,:)>(ratio*2*pi*c*(n/2)),1)-1;
end

if sum(S==0)>0
    L = find(S==0,1)-2;
    S = S(1:(L+1));
end

end