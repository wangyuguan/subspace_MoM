function a_lms = vol_t_vol_coeffs(V, L, S, c)

n = size(V,1);
V = V(:);

if mod(n,2)==0
    x = 2*pi*((-n/2):(n/2-1)); 
else
    x = 2*pi*((-(n-1)/2):((n-1)/2)); 
end



% standard dft grids 
[x,y,z] = meshgrid(x); 
x = x(:); y = y(:); z = z(:);


sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end


nr = ceil(1.5*n); nt = ceil(1.5*n); np = ceil(1.5*n); 
[r,th,phi,w] = spherequad(nr,nt,np,c);
[kx,ky,kz] = Sph2Cart(r,th,phi);
V_nufft = finufft3d3(x,y,z,V(:),-1,eps,kx,ky,kz);

a_lms = zeros(get_num_basis(L,S),1);
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
            a_lms(basis_idx) = (w.*fls.*Ylm)'*V_nufft;
            basis_idx = basis_idx + 1;
        end
    end
end




real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real(real_t_complex\a_lms);
end