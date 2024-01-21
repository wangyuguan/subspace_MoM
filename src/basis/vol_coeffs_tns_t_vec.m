function a_lms = vol_coeffs_tns_t_vec(A_lms, L, S)

a_lms = [];

vec_idx = 1;
for l=0:L
    for s=1:S(l+1)
        for m=-l:l
            a_lms(vec_idx) = A_lms(l+1,s,m+l+1); 
            vec_idx = vec_idx + 1;
        end
    end
end

a_lms = a_lms.';

end