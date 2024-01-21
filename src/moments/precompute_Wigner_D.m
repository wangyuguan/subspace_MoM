function Psi = precompute_Wigner_D(SO3_rule, P)

[real_t_complex,~] = get_view_real_t_complex(P);
vec_idx = @(p, u) p^2+u+p+1; 

nB = size(real_t_complex,1);
nq = size(SO3_rule,1);
Psi = zeros(nq, nB);

for i=1:nq

    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    
    for p=0:P
        Wig_p = wignerD(p, alpha, beta, gamma); 

  
        for u=-p:p
            Psi(i,vec_idx(p,u)) = Wig_p(u+p+1,0+p+1);
        end
    end
end


Psi = Psi*real_t_complex;

end
