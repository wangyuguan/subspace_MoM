function a_lms_rot = rotate_vol_coeffs(a_lms, Rot, L, S)

real_t_complex = get_vol_real_t_complex(L,S);
[alpha, beta, gamma] = rot2elr(Rot);
A_lms = vol_coeffs_vec_t_tns(real_t_complex*a_lms, L, S);

A_lms_rot = zeros(size(A_lms));
for l=0:L
    Dl = wignerD(l,alpha,beta,gamma);
    A_lms_rot(l+1,1:S(l+1),1:(2*l+1)) = tns_mult(A_lms(l+1,1:S(l+1),1:(2*l+1)),3,Dl,2);
end

a_lms_rot = real(real_t_complex\vol_coeffs_tns_t_vec(A_lms_rot, L, S));
end