function [A,b] = create_viewing_constrs(P)
sphere_rule = readtable('sphere_rules/N030_M322_C4.dat');

sphere_rule = sphere_rule{:,:};
alpha = pi-atan2(sphere_rule(:,1),sphere_rule(:,2));
beta = acos(sphere_rule(:,3));
w = sphere_rule(:,4);
q = numel(w);


vec_idx = @(p, u) p^2+u+p+1; 
[real_t_complex,~] = get_view_real_t_complex(P);
Psi = zeros(q,size(real_t_complex,1));
for i=1:q
    for p=0:P
        Wig_p = wignerD(p, alpha(i), beta(i), 1); 
    
    
        for u=-p:p
            Psi(i,vec_idx(p,u)) = Wig_p(u+p+1,0+p+1);
        end
    end
end

Psi = Psi*real_t_complex;
A = -Psi(:,2:end);
b = Psi(:,1);
end