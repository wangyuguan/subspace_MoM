clear 
clc 


load('moms_emp.mat')
load('w_so3.mat')
load('params.mat')
load('moms_emp.mat')
load('constraint.mat')
load('precomps.mat')


% na = size(Phi_precomps_m2,3);
% 
% a = xtrue(1:na);
% b = xtrue(na+1:end);
% a = a.';
% b = b.';
% b = [1;b];
% 
% PCs = tns_mult(Phi_precomps_m2,3,a,1);
% w = Psi_precomps_m2 * b;
% w = w_so3_m2.'.*w;

% m1 = 0;
% for i=1:numel(w)
%     m1 = m1+w(i)*PCs(i,:).';
% end
% 
% norm(m1(:)-m1_emp(:))/norm(m1_emp(:))
% 
% 
% 
% m2 = 0;
% for i=1:numel(w)
%     I = PCs(i,:).';
%     m2 = m2+w(i)*(I*I');
% end
% 
% 
% norm(m2(:)-m2_emp(:))/norm(m2_emp(:))
% 
% 
% 
% 
% PCs = tns_mult(Phi_precomps_m3,3,a,1);
% w = Psi_precomps_m3 * b;
% w = w_so3_m3.'.*w;
% 
% m3 = 0;
% for i=1:numel(w)
%     I = PCs(i,:).';
%     m3 = m3+w(i)*tns_kron3(I,I,I);
% end
% 
% 
% norm(m3(:)-m3_emp(:))/norm(m3_emp(:))





Phi_lms_nodes_MoMs.m2 = {};
for i=1:size(Phi_precomps_m2,1)
    Phi_lms_nodes_MoMs.m2{i} = squeeze(Phi_precomps_m2(i,:,:));
end


Phi_lms_nodes_MoMs.m3 = {};
for i=1:size(Phi_precomps_m3,1)
    Phi_lms_nodes_MoMs.m3{i} = squeeze(Phi_precomps_m3(i,:,:));
end



Psi_MoMs.m2 = Psi_precomps_m2;
Psi_MoMs.m3 = Psi_precomps_m3;



SO3_rule_MoMs.m2 = w_so3_m2(:);
SO3_rule_MoMs.m3 = w_so3_m3(:);


l1 = 1/norm(m1_emp(:))^2;
l2 = 1/norm(m2_emp(:))^2;
l3 = 1/norm(m3_emp(:))^2;

tic 
[cost,grad] = find_cost_grad(x0(:), m1_emp, m2_emp, m3_emp, l1, l2, l3, Phi_lms_nodes_MoMs, Psi_MoMs, SO3_rule_MoMs);
toc 

options = optimoptions('fmincon','Algorithm','sqp','Display','iter', ...
    'SpecifyObjectiveGradient',true,...
    'StepTolerance',1e-6,'FunctionTolerance',1e-6,'StepTolerance',1e-6,...
    'OptimalityTolerance',1e-6,'MaxFunctionEvaluations',1e4,'MaxIterations',1e4);

fun = @(x) find_cost_grad(x, m1_emp, m2_emp, m3_emp, l1, l2, 0, Phi_lms_nodes_MoMs, Psi_MoMs, SO3_rule_MoMs);


b_constr = rhs(:);
[x, xcost] = fmincon(fun,0*x0(:),A_constr,b_constr,[],[],[],[],[],options);