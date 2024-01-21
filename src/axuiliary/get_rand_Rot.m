function Rot = get_rand_Rot
alpha = rand*2*pi;
gamma = rand*2*pi;
beta = acos(1-2*rand);
Rot = Rz(alpha)*Ry(beta)*Rz(gamma); 
end


function Rot = Ry(a)
Rot = [cos(a) 0 sin(a); 
    0 1 0;
    -sin(a) 0 cos(a)];
end

function Rot = Rz(a)
Rot = [cos(a) -sin(a) 0; 
    sin(a) cos(a) 0;
    0 0 1];
end