function [MoMs, bases, evals, M1, Y2, Y3, t_sketch, t_form] = vol_t_real_images_t_emp_MoMs(V, Rots, s, eta2, eta3, fixed_ranks, noise_std)

rng(1)

N = size(Rots,3);
n = size(V,1);
batch_size = 1e3;
num_streams = ceil(N/batch_size);
c = 0.5;

% real grids 
if mod(n,2)==0
    x = 2*pi*((-n/2):(n/2-1)); 
else
    x = 2*pi*((-(n-1)/2):((n-1)/2)); 
end
[x3d,y3d,z3d] = meshgrid(x);
x3d = x3d(:); y3d = y3d(:); z3d = z3d(:);

[x2d,y2d] = meshgrid(x);
x2d = x2d(:); y2d = y2d(:);

% frequency grids 
nr = ceil(1.2*n); nph = ceil(1.6*n);
[r, wr] = lgwt(nr,0,c);
r = flip(r); wr = flip(wr);
[wph, phi] = circle_rule(nph);
wph = wph*2*pi;
[r,phi] = meshgrid(r,phi);
r = r(:); phi = phi(:);
[wr, wph] = meshgrid(wr, wph);
wr = wr(:); wph = wph(:);
w = wr.*wph.*r;

kx = r.*cos(phi); 
ky = r.*sin(phi);

t_sketch = 0;
t_form = 0;


if fixed_ranks
    r2 = eta2;
    r3 = eta3;
end


G = randn(n^2,s);
G1 = randn(n^2,s);
G2 = randn(n^2,s);



% skecth the moments 
M1 = 0;
Y2 = 0;
Y3 = 0;
rng(1)
for i=1:num_streams
    disp(['sketching the moments over the stream No.',num2str(i), '/', num2str(num_streams)])
    tic
    rots = Rots(:,:,((i-1)*batch_size+1):min(i*batch_size,N));
    projections = vol_t_real_images(V,rots,x3d,y3d,z3d,x2d,y2d,kx,ky,w);

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
    rots = Rots(:,:,((i-1)*batch_size+1):min(i*batch_size,N));
    projections = vol_t_real_images(V,rots,x3d,y3d,z3d,x2d,y2d,kx,ky,w);
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



function projections = vol_t_real_images(V,rots,x3d,y3d,z3d,x2d,y2d,kx,ky,w)
warning('off')
accuracy = 1e-6;
N = size(rots,3);
projections = cell(1,N);
n = size(V,1);
parfor j=1:N
    R = rots(:,:,j)';
    grids = R(:,1)*kx'+R(:,2)*ky';
    s = grids(1,:)';
    t = grids(2,:)';
    u = grids(3,:)';
    IF_rot = finufft3d3(x3d,y3d,z3d,V,-1,accuracy,s,t,u);
    projections{j} = real(finufft2d3(kx,ky,IF_rot.*w,1,accuracy,x2d,y2d));
end 
end


function [M1, Y2, Y3] = projs_t_sketched_MoMs(projections, M1, Y2, Y3, G, G1, G2, N)

% K = size(projections,3);
K = numel(projections);

for k=1:K

IR = projections{k};

M1 = M1 + IR/N;
Y2 = Y2 + IR*(IR'*G)/N;
Y3 = Y3 + IR*((G1'*IR).*(G2'*IR)).'/N;

end
    
    
end



function [m2, m3] = projs_t_subspace_MoMs(projections, m2, m3, U2, U3, N)

K = numel(projections);

for k=1:K

IR = projections{k};

I2 = U2'*IR;
I3 = U3'*IR;

m2 = m2 + (I2*I2')/N;
m3 = m3 + tns_kron(tns_kron(I3,I3),I3)/N;

end


end


function U = bases_transform(U)
[d,r] = size(U);
n = sqrt(d);
for i=1:r
    u = reshape(U(:,i),[n,n]);
    u = fftshift(fft2(ifftshift(u)))/n;
    U(:,i) = u(:);
end
end