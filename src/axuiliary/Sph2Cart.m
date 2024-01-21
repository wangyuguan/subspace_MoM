function [x,y,z] = Sph2Cart(r,th,phi)
x = r.*sin(th).*cos(phi);
y = r.*sin(th).*sin(phi);
z = r.*cos(th);
end