function [f,x,y,z,w] = plot_view_density(b_pu, P, Rot, reflect)

sphere_rule = readtable('sphere_rules/N044_M672_Ico.dat');

sphere_rule = sphere_rule{:,:};
alpha = pi-atan2(sphere_rule(:,1),sphere_rule(:,2));
beta = acos(sphere_rule(:,3));
if reflect
    beta = pi-beta;
end
w = sphere_rule(:,4);
q = numel(w);


x = zeros(q,1);
y = zeros(q,1);
z = zeros(q,1);
f = zeros(q,1);

for i=1:q
    x(i) = sin(beta(i))*cos(alpha(i));
    y(i) = sin(beta(i))*sin(alpha(i));
    z(i) = cos(beta(i));
    R = elr2rot(alpha(i),beta(i),1);
    f(i) = get_SO3_density(Rot*R, b_pu, P);
end


F = scatteredInterpolant(x,y,z,f);
[X,Y,Z] = sphere(50);
C = F(X,Y,Z);


surf(X,Y,Z,C)
shading interp;
c = colorbar;
set(c,'Fontsize', 15)

end