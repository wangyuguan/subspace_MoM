function  [moments, bases, svals] = form_analytical_moments(a_lms, b_pu, sampling_size, eta2, eta3, L, S, P, n, c, ranks_provided)

rng(1)
d = n^2;

if ranks_provided
    r2 = eta2;
    r3 = eta3;
end

% prepare the volume coefficients
real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real_t_complex*a_lms;
A_lms_tns = vol_coeffs_vec_t_tns(a_lms, L, S);


sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end
if mod(n,2)==0
    k = ((-n/2):(n/2-1))/n;  
else
    k = ((-(n-1)/2):((n-1)/2))/n; 
end

[kx,ky] = meshgrid(k);
kx = kx(:); ky = ky(:); 


% form basis matrix
grids = [kx';ky';zeros(size(ky'))]; 
grids = grids';
[r,th,phi] = Cart2Sph(grids(:,1), grids(:,2), grids(:,3));


n_vol = get_num_basis(L,S);
base_mat = zeros(n^2,n_vol);
idx = 1;

for l=0:L

    Yl = sph_harmonics(l, th, phi);
    zl = sphbes_zeros(l+1,:);
    cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));

    fl = sphbes(l, r*zl/c, false)*diag(cl); fl(r>c)=0;
    for s=1:S(l+1)
        fls = fl(:,s);
        for m=-l:l
            Ylm = Yl(:,m+l+1);
            base_mat(:,idx) = (fls.*Ylm);
            idx = idx + 1;
        end
    end
end


% first moment
SO3_rule = get_SO3_rule(L,P,1);
q = size(SO3_rule,1);
Psi = precompute_Wigner_D(SO3_rule, P);
rho = Psi*[1;b_pu];
w = SO3_rule(:,4).*rho;

M1 = 0; 
for i=1:q
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    Rot = elr2rot(alpha, beta, gamma);
    IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot);
    M1 = M1 + w(i)*IR;
end




% second moment
SO3_rule = get_SO3_rule(L,P,2);
q = size(SO3_rule,1);
Psi = precompute_Wigner_D(SO3_rule, P);
rho = Psi*[1;b_pu];
w = SO3_rule(:,4).*rho;

Y2 = 0;
G = randn(d, sampling_size);
for i=1:q
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    Rot = elr2rot(alpha, beta, gamma);
    IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot);
    Y2 = Y2 + w(i)*IR*(IR'*G);
end

[U2,S2,~] = svd(Y2, 'econ'); 
s2 = diag(S2);
svals.m2 = s2;
if ~ranks_provided
    r2 = find(s2/s2(1) < eta2, 1,  'first');
end
U2 = U2(:,1:r2);

m2 = 0;
for i=1:q
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    Rot = elr2rot(alpha, beta, gamma);
    IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot);
    IR = U2'*IR;
    m2 = m2 + w(i)*(IR*IR');
end

moments.m2 = m2;
bases.U2 = U2;

moments.m1 = M1;
bases.U1 = eye(d);


% third moment 
SO3_rule = get_SO3_rule(L,P,3);
q = size(SO3_rule,1);
Psi = precompute_Wigner_D(SO3_rule, P);
rho = Psi*[1;b_pu];
w = SO3_rule(:,4).*rho;

Y3 = 0;
G1 = randn(d, sampling_size);
G2 = randn(d, sampling_size);
for i=1:q
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    Rot = elr2rot(alpha, beta, gamma);
    IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot);
    Y3 = Y3 + w(i)*IR*((G1'*IR).*(G2'*IR)).';
end

[U3,S3,~] = svd(Y3, 'econ'); 
s3 = diag(S3);
svals.m3 = s3;
if ~ranks_provided
    r3 = find(s3/s3(1) < eta3, 1,  'first');
end
U3 = U3(:,1:r3);

m3 = 0;
for i=1:q
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);
    Rot = elr2rot(alpha, beta, gamma);
    IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot);
    IR = U3'*IR;
    m3 = m3 + w(i)*tns_kron(tns_kron(IR,IR),IR);
end

moments.m3 = m3;
bases.U3 = U3;

end




function IR = get_2D_projection(A_lms_tns, L, S, base_mat, Rot)



[alpha, beta, gamma] = rot2elr(Rot);

A_lms_tns_rot = zeros(size(A_lms_tns));
for l=0:L
    Dl = wignerD(l,alpha,beta,gamma);
    A_lms_tns_rot(l+1,1:S(l+1),1:(2*l+1)) = tns_mult(A_lms_tns(l+1,1:S(l+1),1:(2*l+1)),3,Dl,2);
end

A_lms_rot = vol_coeffs_tns_t_vec(A_lms_tns_rot, L, S);
IR = base_mat*A_lms_rot;


end

















% function  [moments, bases, svals] = form_analytical_moments(a_lms, b_pu, sampling_size, r2, r3, L, S, P, n, c)
% 
% rng(1)
% d = n^2;
% 
% real_t_complex = get_vol_real_t_complex(L,S);
% A_lms = real_t_complex*a_lms;
% 
% 
% sphbes_zeros = zeros([L+1,max(S)]);
% for l=0:L
%     sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
% end
% if mod(n,2)==0
%     k = ((-n/2):(n/2-1))/n;  
% else
%     k = ((-(n-1)/2):((n-1)/2))/n; 
% end
% 
% [kx,ky] = meshgrid(k);
% kx = kx(:); ky = ky(:); 
% 
% 
% % first moment
% SO3_rule = get_SO3_rule(L,P,1);
% q = size(SO3_rule,1);
% Psi = precompute_Wigner_D(SO3_rule, P);
% rho = Psi*[1;b_pu];
% w = SO3_rule(:,4).*rho;
% 
% M1 = 0; 
% for i=1:q
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
%     Rot = elr2rot(alpha, beta, gamma);
%     IR = vol_coeffs_vec_t_2D_projection(A_lms, L, S, c, sphbes_zeros, kx, ky, Rot);
%     M1 = M1 + w(i)*IR;
% end
% 
% 
% 
% 
% % second moment
% SO3_rule = get_SO3_rule(L,P,2);
% q = size(SO3_rule,1);
% Psi = precompute_Wigner_D(SO3_rule, P);
% rho = Psi*[1;b_pu];
% w = SO3_rule(:,4).*rho;
% 
% Y2 = 0;
% G = randn(d, sampling_size);
% for i=1:q
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
%     Rot = elr2rot(alpha, beta, gamma);
%     IR = vol_coeffs_vec_t_2D_projection(A_lms, L, S, c, sphbes_zeros, kx, ky, Rot);
%     Y2 = Y2 + w(i)*IR*(IR'*G);
% end
% 
% [U2,S2,~] = svd(Y2, 'econ'); 
% s2 = diag(S2);
% svals.m2 = s2;
% U2 = U2(:,1:r2);
% 
% m2 = 0;
% for i=1:q
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
%     Rot = elr2rot(alpha, beta, gamma);
%     IR = vol_coeffs_vec_t_2D_projection(A_lms, L, S, c, sphbes_zeros, kx, ky, Rot);
%     IR = U2'*IR;
%     m2 = m2 + w(i)*(IR*IR');
% end
% 
% moments.m2 = m2;
% bases.U2 = U2;
% 
% moments.m1 = M1;
% bases.U1 = eye(d);
% 
% 
% % third moment 
% SO3_rule = get_SO3_rule(L,P,3);
% q = size(SO3_rule,1);
% Psi = precompute_Wigner_D(SO3_rule, P);
% rho = Psi*[1;b_pu];
% w = SO3_rule(:,4).*rho;
% 
% Y3 = 0;
% G1 = randn(d, sampling_size);
% G2 = randn(d, sampling_size);
% for i=1:q
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
%     Rot = elr2rot(alpha, beta, gamma);
%     IR = vol_coeffs_vec_t_2D_projection(A_lms, L, S, c, sphbes_zeros, kx, ky, Rot);
%     Y3 = Y3 + w(i)*IR*((G1'*IR).*(G2'*IR)).';
% end
% 
% [U3,S3,~] = svd(Y3, 'econ'); 
% s3 = diag(S3);
% svals.m3 = s3;
% U3 = U3(:,1:r3);
% 
% m3 = 0;
% for i=1:q
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
%     Rot = elr2rot(alpha, beta, gamma);
%     IR = vol_coeffs_vec_t_2D_projection(A_lms, L, S, c, sphbes_zeros, kx, ky, Rot);
%     IR = U3'*IR;
%     m3 = m3 + w(i)*tns_kron(tns_kron(IR,IR),IR);
% end
% 
% moments.m3 = m3;
% bases.U3 = U3;
% 
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
