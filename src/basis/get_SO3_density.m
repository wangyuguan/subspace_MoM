function mu = get_SO3_density(Rot, b_pu, P)
vec_idx = @(p, u) p^2+u+p+1; 
[alpha, beta, gamma] = rot2elr(Rot);
[real_t_complex,~] = get_view_real_t_complex(P);
B_pu = real_t_complex*[1; b_pu];
Psi = zeros(numel(B_pu),1);
for p=0:P
    Wig_p = wignerD(p, alpha, beta, gamma); 


    for u=-p:p
        Psi(vec_idx(p,u)) = Wig_p(u+p+1,0+p+1);
    end
end
mu = real(sum(Psi.*B_pu));
end