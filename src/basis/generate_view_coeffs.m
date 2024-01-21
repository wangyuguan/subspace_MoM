function b_pu = generate_view_coeffs(mus,w,k,P)

if k==0
    func = @(alpha, beta, gamma) 1;
else
    func = @(alpha, beta, gamma) 4*pi*vonMisesFisher3D(mus,w,[sin(beta)*cos(alpha);sin(beta)*sin(alpha);cos(beta)],k);
end

B_puv = WignerD_transform(func, P);
B_puv = B_puv/B_puv(1);

B_pu = B_puv(find_view_support(P));



[~, complex_t_real] = get_view_real_t_complex(P);
b_pu = complex_t_real*B_pu;
b_pu = b_pu(2:end);
end




function B_puv = WignerD_transform(func, P)

alpha = pi/(P+1)*(0:(2*P+1));
gamma = pi/(P+1)*(0:(2*P+1));

h_alpha = alpha(2)-alpha(1);
w1 = ones([1, 2*(P+1)])*h_alpha;
h_gamma = gamma(2)-gamma(1);
w3 = ones([1, 2*(P+1)])*h_gamma;

[x,w2]=lgwt(2*(P+1),-1,1);

x = flip(x);
w2 = flip(w2);

B_puv = [];


for l=0:P

    temp = zeros([2*l+1, 2*l+1]);
    
    for j1=1:2*(P+1)
        for k=1:2*(P+1)
            for j2=1:2*(P+1)

                [D_l, ~, ~] = wignerD(l,alpha(j1), acos(x(k)), gamma(j2));
                temp = temp + (2*l+1)*(w1(j1)*w2(k)*w3(j2))*func(alpha(j1), acos(x(k)), gamma(j2))*conj(D_l)/(8*pi^2);

            end
        end
    end

    temp = temp.';
    B_puv = [B_puv; temp(:)];

end

end


function idx = find_view_support(P)
idx = [];
count = 1;
for p=0:P
    for u=-p:p
        for v=-p:p
            if v==0
                idx = [idx; count];
            end
            count = count+1;
        end
    end
end
end


