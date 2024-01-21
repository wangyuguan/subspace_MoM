function  n_vol = get_num_basis(L,S)
n_vol = 0;
for l=0:L
    for s=1:S(l+1)
        for m=-l:l
            n_vol = n_vol+1;
        end
    end
end
end