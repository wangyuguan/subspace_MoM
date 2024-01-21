function [cost, grad] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, Phi_lms_nodes, Psi, SO3_rule)

d3 = size(m3_est,1);
q = size(SO3_rule,1);
w = SO3_rule(:,4).*(Psi*[1;b_pu]);

PCs = cell(1,q);
for i=1:q
    PCs{i} = Phi_lms_nodes{i}*a_lms; % 9.0%
end


m3 = 0;
PC_ac_cell = cell(q,1);
for i=1:q
    PC = PCs{i};
    PC_ac = tns_kron(tns_kron(PC,PC),PC); % third most expensive 20.3% 
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

    grad_a = grad_a + w(i)*6*real(PC_lms.'*tmp); % second most expensive 29.9%
    grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC3.*PC_ac_cell{i},"all")); % most expensive 31%
end

grad_b = real(Psi'*grad_rho);
grad = [grad_a; grad_b(2:end)];

end


% function [cost, grad] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, PCs_tns, PCs_cell, Psi, SO3_rule)
% 
% PCs = tns_mult(PCs_tns,3,a_lms,1);
% d3 = size(PCs_tns,2);
% q = size(SO3_rule,1);
% w = SO3_rule(:,4).*(Psi*[1;b_pu]);
% 
% m3 = 0;
% PC_ac_cell = cell(q,1);
% for i=1:q
%     PC = PCs(i,:).';
%     PC_ac = tns_kron(tns_kron(PC,PC),PC);
%     PC_ac_cell{i} = PC_ac;
%     m3 = m3 + w(i)*PC_ac;
% end
% 
% C3 = m3-m3_est; cC3 = conj(C3);
% cost = norm(C3(:))^2;
% 
% 
% grad_a = 0;
% grad_rho = zeros(q,1);
% for i=1:q
%     PC = PCs(i,:).';
%     PC_lms = PCs_cell{i};
%     tmp = PC*PC.';
% 
%     tmp = reshape(cC3, [d3,d3^2])*tmp(:);
% 
%     grad_a = grad_a + w(i)*6*real(PC_lms.'*tmp);
%     grad_rho(i) = 2*SO3_rule(i,4)*real(sum(cC3.*PC_ac_cell{i},"all"));
% end
% 
% grad_b = real(Psi'*grad_rho);
% grad = [grad_a; grad_b(2:end)];
% 
% end