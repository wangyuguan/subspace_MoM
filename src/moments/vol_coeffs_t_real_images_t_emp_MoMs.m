function [MoMs, bases, evals, M1, Y2, Y3, t_sketch, t_form] = vol_coeffs_t_real_images_t_emp_MoMs(a_lms, L, S, n, c, Rots, s, eta2, eta3, fixed_ranks, noise_std)

rng(1)

N = size(Rots,3);
num_streams = ceil(N/1e4);

t_sketch = 0;
t_form = 0;


if fixed_ranks
    r2 = eta2;
    r3 = eta3;
end


G = randn(n^2,s);
G1 = randn(n^2,s);
G2 = randn(n^2,s);


Phi = precompute_real_image_mapping(n, c, L, S);

% prepare the volume coefficients
real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real_t_complex*a_lms;
A_lms_tns = vol_coeffs_vec_t_tns(a_lms, L, S);

% skecth the moments 
M1 = 0;
Y2 = 0;
Y3 = 0;
rng(1)
for i=1:num_streams
    disp(['sketching the moments over the stream No.',num2str(i), '/', num2str(num_streams)])
    tic
    rots = Rots(:,:,((i-1)*1e4+1):min(i*1e4,N));
    projections = view_coeffs_tns_t_2D_real_images(A_lms_tns, L, S, Phi, rots);
    

    % add noise 
    for j=1:numel(projections)
        projections{j} = projections{j} + noise_std*randn(n^2,1);
    end
    toc 

    tic 
    [M1, Y2, Y3] = projs_t_sketched_MoMs(projections, M1, Y2, Y3, G, G1, G2, N);
    t=toc

    t_sketch = t_sketch+t;
end




% debias
Y2 = Y2 - noise_std^2*G;

I = eye(n^2);
for i=1:n^2
    ei = I(:,i);
    Y3 = Y3 - noise_std^2*( M1*((G1'*ei).*(G2'*ei)).' + ei*((G1'*M1).*(G2'*ei)).' + ei*((G1'*ei).*(G2'*M1)).');
end







% obtain subspaces by SVD
[U2,S2,~] = svd(Y2,'econ'); 
s2 = diag(S2);
if ~fixed_ranks
    r2 = find(s2/s2(1) < eta2, 1,  'first');
end
U2 = U2(:,1:r2);
[U3,S3,~] = svd(Y3,'econ');
s3 = diag(S3);
if ~fixed_ranks
    r3 = find(s3/s3(1) < eta3, 1,  'first');
end
U3 = U3(:,1:r3);

evals.m2 = s2; evals.m3 = s3; 






% form the subspace second and third MoMs
m2 = 0;
m3 = 0;
rng(1)
for i=1:num_streams
    disp(['forming the moments over the stream No.',num2str(i), '/', num2str(num_streams)])
    tic 
    rots = Rots(:,:,((i-1)*1e4+1):min(i*1e4,N));
    projections = view_coeffs_tns_t_2D_real_images(A_lms_tns, L, S, Phi, rots);
    % add noise 
    for j=1:numel(projections)
        projections{j} = projections{j} + noise_std*randn(n^2,1);
    end
    toc 

    tic 
    [m2, m3] = projs_t_subspace_MoMs(projections, m2, m3, U2, U3, N);
    t=toc
    t_form = t_form+t;
end



% debias
m2 = m2 - noise_std^2*eye(r2);

I = eye(n^2);
U3M1 = U3'*M1; 
for i=1:n^2
    ei = U3'*I(:,i);
    m3 = m3 - noise_std^2*(tns_kron(tns_kron(U3M1,ei),ei)+tns_kron(tns_kron(ei,U3M1),ei)+tns_kron(tns_kron(ei,ei),U3M1));
end





MoMs.m1 = n*U2'*M1;
MoMs.m2 = n^2*m2;
MoMs.m3 = n^3*m3;

bases.U1 = bases_transform(U2);
bases.U2 = bases_transform(U2);
bases.U3 = bases_transform(U3);



end
















