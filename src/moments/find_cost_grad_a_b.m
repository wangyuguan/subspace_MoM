function [cost,grad] = find_cost_grad_a_b(x, m1_est, m2_est, m3_est, l1, l2, l3, Phi_lms_nodes_MoMs, Psi_MoMs, SO3_rule_MoMs)

cost = 0; grad = 0;
n_vol = size(Phi_lms_nodes_MoMs.m1{1},2);
a_lms = x(1:n_vol);
b_pu = x((n_vol+1):end);



% cost and grad from the first subspace moment
if l1>0
    [cost1, grad1] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, Phi_lms_nodes_MoMs.m1, Psi_MoMs.m1, SO3_rule_MoMs.m1);
    cost = cost + l1*cost1;
    grad = grad + l1*grad1;
end


% cost and grad from the second subspace moment
if l2>0
    [cost2, grad2] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, Phi_lms_nodes_MoMs.m2, Psi_MoMs.m2, SO3_rule_MoMs.m2);
    cost = cost + l2*cost2;
    grad = grad+ l2*grad2;
end



% cost and grad from the third subspace moment
if l3>0
    [cost3, grad3] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, Phi_lms_nodes_MoMs.m3, Psi_MoMs.m3, SO3_rule_MoMs.m3);
    cost = cost + l3*cost3;
    grad = grad+ l3*grad3;
end

end




% function [cost,grad] = find_cost_grad_a_b(x, m1_est, m2_est, m3_est, l1, l2, l3, PCs_tnss, PCs_cells, Psis, SO3_rules)
% 
% cost = 0; grad = 0;
% n_vol = size(PCs_tnss.m1,3);
% a_lms = x(1:n_vol);
% b_pu = x((n_vol+1):end);
% 
% 
% % cost and grad from the first subspace moment
% if l1>0
%     [cost1, grad1] = form_m1_cost_grad_a_b(a_lms, b_pu, m1_est, PCs_tnss.m1, PCs_cells.m1, Psis.m1, SO3_rules.m1);
%     cost = cost + l1*cost1;
%     grad = grad + l1*grad1;
% end
% 
% 
% % cost and grad from the second subspace moment
% if l2>0
%     [cost2, grad2] = form_m2_cost_grad_a_b(a_lms, b_pu, m2_est, PCs_tnss.m2, PCs_cells.m2, Psis.m2, SO3_rules.m2);
%     cost = cost + l2*cost2;
%     grad = grad+ l2*grad2;
% end
% 
% 
% 
% % cost and grad from the third subspace moment
% if l3>0
%     [cost3, grad3] = form_m3_cost_grad_a_b(a_lms, b_pu, m3_est, PCs_tnss.m3, PCs_cells.m3, Psis.m3, SO3_rules.m3);
%     cost = cost + l3*cost3;
%     grad = grad+ l3*grad3;
% end
% 
% end

