function I = vol_coeffs_t_Fourier_2D_proj(a_lms, L, S, n, c)

sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end

% form base matrix

if mod(n,2)==0
    k = ((-n/2):(n/2-1))/n;  
else
    k = ((-(n-1)/2):((n-1)/2))/n; 
end

[kx,ky] = meshgrid(k);
kx = kx(:); 
ky = ky(:);
grids = [kx';ky';zeros(size(ky'))]; 
grids = grids';
[r,th,phi] = Cart2Sph(grids(:,1), grids(:,2), grids(:,3));


n_vol = get_num_basis(L,S);
base_3D_mat = zeros(n^2,n_vol);
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
            base_3D_mat(:,idx) = (fls.*Ylm);
            idx = idx + 1;
        end
    end
end


real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real_t_complex*a_lms;


I = reshape(base_3D_mat*a_lms,[n,n]);

end