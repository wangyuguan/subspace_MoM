function [cost, grad] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, Phi_lms_nodes, Psi, SO3_rule)


q = size(SO3_rule,1);
w = SO3_rule(:,4).*(Psi*[1;b_pu]);


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
    grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC2.*PC_ac_cell{i},"all"));
end
grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];

end


% function [cost, grad] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, PCs_tns, PCs_cell, Psi, SO3_rule)
% 
% PCs = tns_mult(PCs_tns,3,a_lms,1);
% 
% q = size(SO3_rule,1);
% w = SO3_rule(:,4).*(Psi*[1;b_pu]);
% 
% m2 = 0;
% PC_ac_cell = cell(q,1);
% for i=1:q
%     PC = PCs(i,:).';
%     PC_ac = (PC*PC');
%     PC_ac_cell{i} = PC_ac;
%     m2 = m2 + w(i)*PC_ac;
% end
% 
% C2 = m2-m2_est;  cC2 = conj(C2);
% cost = norm(C2(:))^2;
% 
% 
% grad_a = 0;
% grad_rho = zeros(q,1);
% for i=1:q
%     PC = PCs(i,:).';
%     PC_lms = PCs_cell{i};
%     
%     grad_a = grad_a + w(i)*4*real(PC_lms.'*(cC2*conj(PC)));
%     grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC2.*PC_ac_cell{i},"all"));
% end
% grad_b = real(Psi'*grad_rho);
% grad = [grad_a; grad_b(2:end)];
% 
% end