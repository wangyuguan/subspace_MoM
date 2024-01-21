function [Rot,cost,reflect] = view_coeffs_align(b_pu_1, P1, b_pu_2, P2, trials)

[Rot1,cost1] = run_view_coeffs_align(b_pu_1, P1, b_pu_2, P2, true, trials);
[Rot2,cost2] = run_view_coeffs_align(b_pu_1, P1, b_pu_2, P2, false, trials);


if cost1<cost2
    Rot = Rot1;
    cost = cost1;
    reflect = true;
else
    Rot = Rot2;
    cost = cost2;
    reflect = true;
end



end


function [Rot,cost] = run_view_coeffs_align(b_pu_1, P1, b_pu_2, P2, reflect, trials)

sphere_rule = readtable('sphere_rules/N034_M410_C4.dat');
sphere_rule = sphere_rule{:,:};
alpha = pi-atan2(sphere_rule(:,1),sphere_rule(:,2));
beta = acos(sphere_rule(:,3));
w = sphere_rule(:,4);
q = numel(w);

f0 = zeros(q,1);
for i=1:q
    R = elr2rot(alpha(i),beta(i),1);
    f0(i) = get_SO3_density(R, b_pu_1, P1);
end


Rots = cell(1,trials);
for i=1:trials
    Rots{i} = randrot(3);
end

costs = zeros(1,trials);
parfor i=1:trials
    Rot = Rots{i};

    if reflect
        costs(i) = get_cost(Rot, alpha, pi-beta, w, q, b_pu_2, P2, f0);
    else
        costs(i) = get_cost(Rot, alpha, beta, w, q, b_pu_2, P2, f0);
    end
end
[~,i] = min(costs);
Rot0 = Rots{i};


problem.M = rotationsfactory(3, 1);

if reflect
    problem.cost = @(Rot) get_cost(Rot, alpha, pi-beta, w, q, b_pu_2, P2, f0);
else
    problem.cost = @(Rot) get_cost(Rot, alpha, beta, w, q, b_pu_2, P2, f0);
end

[Rot, cost] = conjugategradient(problem,Rot0);

end




function cost = get_cost(Rot, alpha, beta, w, q, b_pu, P, f0)

f = zeros(q,1);
parfor i=1:q
    R = elr2rot(alpha(i),beta(i),1);
    f(i) = get_SO3_density(Rot*R, b_pu, P);
end

cost = w'*(f-f0).^2; 

end