function [Phi_lms_nodes_MoMs, Psi_MoMs] = do_precomputations(L, S, P, bases, n, c, SO3_rule_MoMs)

% first moment
disp('precomputing things related to M1...')
Phi_lms_nodes = precompute_PCs(bases.U1, L, S, n, c, SO3_rule_MoMs.m1);
Phi_lms_nodes_MoMs.m1 = Phi_lms_nodes;
Psi_MoMs.m1 = precompute_Wigner_D(SO3_rule_MoMs.m1, P);


% second moment 
disp('precomputing things related to M2...')
Phi_lms_nodes = precompute_PCs(bases.U2, L, S, n, c, SO3_rule_MoMs.m2);
Phi_lms_nodes_MoMs.m2 = Phi_lms_nodes;
Psi_MoMs.m2 = precompute_Wigner_D(SO3_rule_MoMs.m2, P);


% third moment 
disp('precomputing things related to M3...')
Phi_lms_nodes = precompute_PCs(bases.U3, L, S, n, c, SO3_rule_MoMs.m3);
Phi_lms_nodes_MoMs.m3 = Phi_lms_nodes;
Psi_MoMs.m3 = precompute_Wigner_D(SO3_rule_MoMs.m3, P);


end





% function [PCs_tnss, PCs_cells, Psis] = do_precomputations(L, S, P, bases, n, c, SO3_rules)
% 
% % first moment
% [PCs_tns, PCs_cell] = precompute_PCs(bases.U1, L, S, n, c, SO3_rules.m1);
% PCs_tnss.m1 = PCs_tns;
% PCs_cells.m1 =  PCs_cell; 
% Psis.m1 = precompute_Wigner_D(SO3_rules.m1, P);
% 
% 
% % second moment 
% [PCs_tns, PCs_cell] = precompute_PCs(bases.U2, L, S, n, c, SO3_rules.m2);
% PCs_tnss.m2 = PCs_tns;
% PCs_cells.m2 =  PCs_cell; 
% Psis.m2 = precompute_Wigner_D(SO3_rules.m2, P);
% 
% 
% % third moment 
% [PCs_tns, PCs_cell] = precompute_PCs(bases.U3, L, S, n, c, SO3_rules.m3);
% PCs_tnss.m3 = PCs_tns;
% PCs_cells.m3 =  PCs_cell; 
% Psis.m3 = precompute_Wigner_D(SO3_rules.m3, P);
% 
% 
% end




