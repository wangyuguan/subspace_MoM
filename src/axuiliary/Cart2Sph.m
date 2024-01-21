function [r,th,phi] = Cart2Sph(x,y,z)
% r = sqrt(x.^2+y.^2+z.^2);
% th = acos(z./r);
% phi = sign(y).*acos(x./sqrt(x.^2+y.^2));
% phi(phi<0) = phi(phi<0)+2*pi;
% th(isnan(th)) = 0;
% phi(isnan(phi)) = 0;


[phi,th,r] = cart2sph(x,y,z);
phi(phi<0) = phi(phi<0)+2*pi;
th = pi/2-th;
end