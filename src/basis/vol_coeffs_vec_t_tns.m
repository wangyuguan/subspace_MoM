function A_lms = vol_coeffs_vec_t_tns(a_lms, L, S)

A_lms = zeros(L+1, max(S), 2*L+1);

vec_idx = 1;
for l=0:L
    for s=1:S(l+1)
        for m=-l:l
            A_lms(l+1,s,m+l+1) = a_lms(vec_idx);
            vec_idx = vec_idx + 1;
        end
    end
end

end