function I = vol_coeffs_t_real_2D_proj(a_lms, L, S, n, c)

if mod(n,2)==0
    x = 2*pi*((-n/2):(n/2-1)); 
else
    x = 2*pi*((-(n-1)/2):((n-1)/2)); 
end
[x,y] = meshgrid(x);
x = x(:); y = y(:); 

sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end


nr = ceil(2*n); nph = ceil(2*n);
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

Phi = [];
for l=0:L
    Yl = sph_harmonics(l, pi/2*ones(size(phi)), phi);
    zl = sphbes_zeros(l+1,:);
    cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));

    fl = sphbes(l, r*zl/c, false)*diag(cl); fl(r>c)=0;
    for s=1:S(l+1)
        fls = fl(:,s);
        for m=-l:l
            Ylm = Yl(:,m+l+1);
            Phi =[Phi,finufft2d3(kx,ky,w.*fls.*Ylm,1,eps,x,y)];
        end
    end
end


real_t_complex = get_vol_real_t_complex(L,S);
a_lms = real_t_complex*a_lms;


I = real(reshape(Phi*a_lms,[n,n]));

end