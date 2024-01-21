function [cost, grad] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, Phi_lms_nodes, Psi, SO3_rule)


q = size(SO3_rule,1);
w = SO3_rule(:,4).*(Psi*[1;b_pu]);

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
    grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC1.*PC,"all"));
end

grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];

end




% function [cost, grad] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, PCs_tns, PCs_cell, Psi, SO3_rule)
% 
% PCs = tns_mult(PCs_tns,3,a_lms,1);
% 
% q = size(SO3_rule,1);
% w = SO3_rule(:,4).*(Psi*[1;b_pu]);
% 
% m1 = 0;
% 
% for i=1:q
%     m1 = m1 + w(i)*PCs(i,:).';
% end
% 
% C1 = m1-m1_est;  cC1 = conj(C1);
% cost = norm(C1(:))^2;
% 
% 
% grad_a = 0;
% grad_rho = zeros(q,1);
% for i=1:q
%     PC = PCs(i,:).';
%     PC_lms = PCs_cell{i};
%     
%     grad_a = grad_a + w(i)*2*real(PC_lms'*C1);
%     grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC1.*PC,"all"));
% end
% 
% grad_b = real(Psi'*grad_rho);
% grad = [grad_a; grad_b(2:end)];
% 
% end
