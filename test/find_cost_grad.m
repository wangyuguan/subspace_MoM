function [cost,grad] = find_cost_grad(x, m1_est, m2_est, m3_est, l1, l2, l3, Phi_lms_nodes_MoMs, Psi_MoMs, SO3_rule_MoMs)

cost = 0; grad = 0;
n_vol = size(Phi_lms_nodes_MoMs.m2{1},2);
a_lms = x(1:n_vol);
b_pu = x((n_vol+1):end);



% cost and grad from the first subspace moment
if l1>0
    % [cost1, grad1] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, Phi_lms_nodes_MoMs.m1, Psi_MoMs.m1, SO3_rule_MoMs.m1);
    [cost1, grad1] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, Phi_lms_nodes_MoMs.m2, Psi_MoMs.m2, SO3_rule_MoMs.m2);
    cost = cost + l1*cost1;
    grad = grad + l1*grad1;
end


% cost and grad from the second subspace moment
if l2>0
    [cost2, grad2] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, Phi_lms_nodes_MoMs.m2, Psi_MoMs.m2, SO3_rule_MoMs.m2);
    cost = cost + l2*cost2;
    grad = grad + l2*grad2;
end



% cost and grad from the third subspace moment
if l3>0
    [cost3, grad3] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, Phi_lms_nodes_MoMs.m3, Psi_MoMs.m3, SO3_rule_MoMs.m3);
    cost = cost + l3*cost3;
    grad = grad + l3*grad3;
end


end
%
%
%
%
%
%
function [cost, grad] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, Phi_lms_nodes, Psi, SO3_rule)


q = size(SO3_rule,1);
w = SO3_rule.*(Psi*[1;b_pu]);


PCs = cell(1,q);
for i=1:q
    PCs{i} = Phi_lms_nodes{i}*a_lms;
end


m1 = 0;

for i=1:q
    m1 = m1 + w(i)*PCs{i};
end

C1 = m1-m1_est;  cC1 = conj(C1);
cost = norm(C1(:))^2;


grad_a = 0;
grad_rho = zeros(q,1);
for i=1:q
    PC = PCs{i};
    PC_lms = Phi_lms_nodes{i};
    
    grad_a = grad_a + w(i)*2*real(PC_lms'*C1);
    grad_rho(i) = 2*SO3_rule(i)*real(sum(cC1.*PC,"all"));
end

grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];

end
%
%
%
%
%
%
function [cost, grad] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, Phi_lms_nodes, Psi, SO3_rule)


q = size(SO3_rule,1);
w = SO3_rule.*(Psi*[1;b_pu]);


PCs = cell(1,q);
for i=1:q
    PCs{i} = Phi_lms_nodes{i}*a_lms;
end


m2 = 0;
PC_ac_cell = cell(q,1);
for i=1:q
    PC = PCs{i};
    PC_ac = (PC*PC');
    PC_ac_cell{i} = PC_ac;
    m2 = m2 + w(i)*PC_ac;
end

C2 = m2-m2_est;  cC2 = conj(C2);
cost = norm(C2(:))^2;


grad_a = 0;
grad_rho = zeros(q,1);
for i=1:q
    PC = PCs{i};
    PC_lms = Phi_lms_nodes{i};
    
    grad_a = grad_a + w(i)*4*real(PC_lms.'*(cC2*conj(PC)));
    grad_rho(i) = 2*SO3_rule(i)*real(sum(cC2.*PC_ac_cell{i},"all"));
end
grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];


end
%
%
%
%
%
%
function [cost, grad] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, Phi_lms_nodes, Psi, SO3_rule)

d3 = size(m3_est,1);
q = size(SO3_rule,1);
w = SO3_rule.*(Psi*[1;b_pu]);

PCs = cell(1,q);
for i=1:q
    PCs{i} = Phi_lms_nodes{i}*a_lms; 
end


m3 = 0;
PC_ac_cell = cell(q,1);
for i=1:q
    PC = PCs{i};
    PC_ac = tns_kron(tns_kron(PC,PC),PC); 
    PC_ac_cell{i} = PC_ac;
    m3 = m3 + w(i)*PC_ac;
end

C3 = m3-m3_est; 
cC3 = conj(C3);
cost = norm(C3(:))^2;


grad_a = 0;
grad_rho = zeros(q,1);
for i=1:q
    PC = PCs{i};
    PC_lms = Phi_lms_nodes{i};
    tmp = PC*PC.';

    tmp = reshape(cC3, [d3,d3^2])*tmp(:);

    grad_a = grad_a + w(i)*6*real(PC_lms.'*tmp); 
    grad_rho(i) = 2*SO3_rule(i)*real(sum(cC3.*PC_ac_cell{i},"all")); 
end

grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];

end