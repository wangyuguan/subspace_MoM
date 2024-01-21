function [cost, a_lms_aligned, reflect] = vol_coeffs_align(a_lms, a_lms_ref, L, S, trials)


[cost1, a_lms_aligned1] = run_vol_coeffs_align(a_lms, a_lms_ref, L, S, trials, true);
[cost2, a_lms_aligned2] = run_vol_coeffs_align(a_lms, a_lms_ref, L, S, trials, false);

if cost1<cost2
    cost = cost1;
    a_lms_aligned = a_lms_aligned1;
    reflect = true;
else
    cost = cost2;
    a_lms_aligned = a_lms_aligned2;
    reflect = false;
end

end



function [cost, a_lms_aligned] = run_vol_coeffs_align(a_lms, a_lms_ref, L, S, trials, reflect)

if reflect
    a_lms = reflect_vol_coeffs(a_lms, 70, L, S, .5);
end

real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real_t_complex*a_lms;
A_lms = vol_coeffs_vec_t_tns(a_lms, L, S);
a_lms_ref = real_t_complex*a_lms_ref;


Rots = cell(1,trials);
for i=1:trials
    Rots{i} = get_rand_Rot;
end

costs = zeros(1,trials);
parfor i=1:trials
    costs(i) =  get_align_cost(A_lms, a_lms_ref, L, S, Rots{i});
end
[~,i] = min(costs);
Rot0 = Rots{i};

problem.M = rotationsfactory(3,1);
problem.cost = @(Rot) get_align_cost(A_lms, a_lms_ref, L, S, Rot);
[Rot, cost] = trustregions(problem, Rot0);

[alpha, beta, gamma] = rot2elr(Rot);
A_lms_aligned = zeros(size(A_lms));
for l=0:L
    Dl = wignerD(l,alpha,beta,gamma);
    A_lms_aligned(l+1,1:S(l+1),1:(2*l+1)) = tns_mult(A_lms(l+1,1:S(l+1),1:(2*l+1)),3,Dl.',2);
end
a_lms_aligned = real(real_t_complex\vol_coeffs_tns_t_vec(A_lms_aligned, L, S));

end




function cost = get_align_cost(A_lms, a_lms_ref, L, S, Rot)

[alpha, beta, gamma] = rot2elr(Rot);

A_lms_gt_rot = zeros(size(A_lms));
for l=0:L
    Dl = wignerD(l,alpha,beta,gamma);
    A_lms_gt_rot(l+1,1:S(l+1),1:(2*l+1)) = tns_mult(A_lms(l+1,1:S(l+1),1:(2*l+1)),3,Dl.',2);
end

a_lms_gt_rot = vol_coeffs_tns_t_vec(A_lms_gt_rot, L, S);
cost = norm(a_lms_gt_rot-a_lms_ref)^2;

end



function a_lms_reflect = reflect_vol_coeffs(a_lms, n, L, S, c)

real_t_complex = get_vol_real_t_complex(L,S);
A_lms = real_t_complex*a_lms;


sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end

nr = ceil(1.2*n); nt = ceil(1.2*n); np = ceil(1.2*n); 
[r,th,phi,w] = spherequad(nr,nt,np,c);
[kx,ky,kz] = Sph2Cart(r,th,phi);
[r_reflect,th_reflect,phi_reflect] = Cart2Sph(kx,ky,-kz);

% get the reflected volume
V_reflect = 0;
basis_idx = 1;
for l=0:L

    Yl = sph_harmonics(l, th_reflect, phi_reflect);
    zl = sphbes_zeros(l+1,:);
    cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));
    fl = sphbes(l, r_reflect*zl/c, false)*diag(cl);

    for s=1:S(l+1)
        fls = fl(:,s);
        for m=-l:l
            Ylm = Yl(:,m+l+1);
            V_reflect = V_reflect + A_lms(basis_idx) *(fls.*Ylm);
            basis_idx = basis_idx + 1;
        end
    end
end

A_lms_reflect = zeros(size(A_lms));
basis_idx = 1;
for l=0:L

    Yl = sph_harmonics(l, th, phi);
    zl = sphbes_zeros(l+1,:);
    cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));
    fl = sphbes(l, r*zl/c, false)*diag(cl);

    for s=1:S(l+1)
        fls = fl(:,s);
        for m=-l:l
            Ylm = Yl(:,m+l+1);
            A_lms_reflect(basis_idx) = (w.*fls.*Ylm)'*V_reflect;
            basis_idx = basis_idx + 1;
        end
    end
end

a_lms_reflect = real(real_t_complex\A_lms_reflect);


end

