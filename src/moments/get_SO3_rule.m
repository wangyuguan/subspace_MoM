function SO3_rule = get_SO3_rule(L,P,k,red)



N = k*L+P;

if nargin == 4
    N = N-red;
end

if N==1
    sphere_rule = readtable('sphere_rules/N001_M2_Inv.dat');
elseif N==2
    sphere_rule = readtable('sphere_rules/N002_M4_Tetra.dat');
elseif N==3
    sphere_rule = readtable('sphere_rules/N003_M6_Octa.dat');
elseif N==4
    sphere_rule = readtable('sphere_rules/N004_M10_C4.dat');
elseif N==5
    sphere_rule = readtable('sphere_rules/N005_M12_Ico.dat');
elseif N==6
    sphere_rule = readtable('sphere_rules/N006_M18_C4.dat');
elseif N==7
    sphere_rule = readtable('sphere_rules/N007_M22_C5.dat');
elseif N==8
    sphere_rule = readtable('sphere_rules/N008_M28_Tetra.dat');
elseif N==9
    sphere_rule = readtable('sphere_rules/N009_M32_Ico.dat');
elseif N==10
    sphere_rule = readtable('sphere_rules/N010_M42_C4.dat');
elseif N==11
    sphere_rule = readtable('sphere_rules/N011_M48_Octa.dat');
elseif N==12
    sphere_rule = readtable('sphere_rules/N012_M58_C4.dat');
elseif N==13
    sphere_rule = readtable('sphere_rules/N013_M64_Inv.dat');
elseif N==14
    sphere_rule = readtable('sphere_rules/N014_M72_Ico.dat');
elseif N==15
    sphere_rule = readtable('sphere_rules/N015_M82_C5.dat');
elseif N==16
    sphere_rule = readtable('sphere_rules/N016_M98_C4.dat');
elseif N==17
    sphere_rule = readtable('sphere_rules/N017_M104_C3.dat');
elseif N==18
    sphere_rule = readtable('sphere_rules/N018_M122_C4.dat');
elseif N==19
    sphere_rule = readtable('sphere_rules/N019_M130_Inv.dat');
elseif N==20
    sphere_rule = readtable('sphere_rules/N020_M148_Tetra.dat');
elseif N==21
    sphere_rule = readtable('sphere_rules/N021_M156_C3.dat');
elseif N==22
    sphere_rule = readtable('sphere_rules/N022_M178_C4.dat');
elseif N==23
    sphere_rule = readtable('sphere_rules/N023_M186_C3.dat');
elseif N==24
    sphere_rule = readtable('sphere_rules/N024_M210_C4.dat');
elseif N==25
    sphere_rule = readtable('sphere_rules/N025_M220_Inv.dat');
elseif N==26
    sphere_rule = readtable('sphere_rules/N026_M244_Tetra.dat');
elseif N==27
    sphere_rule = readtable('sphere_rules/N027_M254_C3.dat');
elseif N==28
    sphere_rule = readtable('sphere_rules/N028_M282_C4.dat');
elseif N==29
    sphere_rule = readtable('sphere_rules/N029_M292_C5.dat');
elseif N==30
    sphere_rule = readtable('sphere_rules/N030_M322_C4.dat');
elseif N==31 || N==32
    sphere_rule = readtable('sphere_rules/N032_M364_Tetra.dat');
elseif N==33 || N==34
    sphere_rule = readtable('sphere_rules/N034_M410_C4.dat');
elseif N==35
    sphere_rule = readtable('sphere_rules/N035_M422_C5.dat');
elseif N==36
    sphere_rule = readtable('sphere_rules/N036_M458_C4.dat');
elseif N==37
    sphere_rule = readtable('sphere_rules/N037_M472_C5.dat');
elseif N==38
    sphere_rule = readtable('sphere_rules/N038_M508_Tetra.dat');
elseif N==39
    sphere_rule = readtable('sphere_rules/N039_M522_C5.dat');
else
    sphere_rule = readtable('sphere_rules/N044_M672_Ico.dat');
end

sphere_rule = sphere_rule{:,:};
alpha = pi-atan2(sphere_rule(:,1),sphere_rule(:,2));
beta = acos(sphere_rule(:,3));
w_sph = sphere_rule(:,4);
n_sph = numel(w_sph);

[w_circ, gamma] = circle_rule(N-P+1);
n_circ = numel(w_circ);

n_SO3 = n_sph*n_circ;
SO3_rule = zeros(n_SO3,4);
for i=1:n_sph

    SO3_rule(((i-1)*n_circ+1):(i*n_circ),1) = alpha(i);
    SO3_rule(((i-1)*n_circ+1):(i*n_circ),2) = beta(i);

    for j=1:n_circ
        SO3_rule(((i-1)*n_circ+j),3) = gamma(j);
        SO3_rule(((i-1)*n_circ+j),4) = w_sph(i)*w_circ(j);
    end
end
    

end