clear all 
clc 
rng(1)

addpath(genpath('src'))
addpath(genpath('thirdparty'))
addpath ~/finufft/matlab % add your finufft path 
run('~/manopt/importmanopt.m') % import manopt


L = 7;  % spherical Bessel expansion for volume 
P = 4;  % spherical harmonics expansion for viewing direction density 
c = 0.5; % bandlimit 

%% get downsampled volume 
Vol = double(ReadMRC('emd_0409.map'));
dsidx = downsample(1:size(Vol,1),3);
n = numel(dsidx);
Vol = double(Vol(dsidx,dsidx,dsidx));

%% perform spherical-Bessel expansion 
[L,S] = get_truncate_limit(L, n, c, 0.8);
a_lms = vol_t_vol_coeffs(Vol, L, S, c);
n_vol = numel(a_lms);
V_gt = vol_coeffs_t_vol(a_lms, n, c, L, S);

%% generate viewing direction density 
n_vMF = 8;
mus = randn(3,n_vMF);
for i=1:n_vMF
    mus(:,i) = mus(:,i)/norm(mus(:,i));
end
wv = ones(1,n_vMF)/n_vMF;
b_pu = generate_view_coeffs(mus,wv,2,P);


%% form subspace moments 
[MoMs, bases, svals] = form_analytical_moments(a_lms, b_pu, 250, 200, 50, L, S, P, n, c, true);

m1_est = MoMs.m1;
m2_est = MoMs.m2;
m3_est = MoMs.m3;

l1 = 1/norm(m1_est(:))^2;
l2 = 1/norm(m2_est(:))^2;
l3 = 1/norm(m3_est(:))^2;

%% precomputatiom before optimization 
SO3_rules = get_inexact_SO3_rules(L,P,.1,.3,.6); % use inexact rules 
disp('doing precomputations....')
[Phi_lms_nodes, Psi] = do_precomputations(L, S, P, bases, n, c, SO3_rules);

%% two-stage optimization 
options = optimoptions('fmincon', 'Display','iter', ...
    'SpecifyObjectiveGradient',true,...
    'MaxIterations',1e4,'StepTolerance',1e-6);

[A_view,b] = create_viewing_constrs(P);
A = sparse([zeros(size(A_view,1),n_vol) A_view]);  % creating constraints 

n_rep = 5;
a_lms_est = cell(1,n_rep);
b_pu_est = cell(1,n_rep);
losses_M2 = zeros(1,n_rep);


tic; 
% stage one 
fun = @(x) find_cost_grad_a_b(x, m1_est, m2_est, m3_est,...
    l1, l2, 0, Phi_lms_nodes, Psi, SO3_rules);
for i=1:n_rep
    a_lms_initial = randn(n_vol,1)*1e-6;
    b_pu_initial = generate_random_view_coeffs(P,10,10);
    x_initial = [a_lms_initial; b_pu_initial];
    [x, xcost] = fmincon(fun,x_initial,A,b,[],[],[],[],[],options);
    a_lms_est{i} = x(1:n_vol);
    b_pu_est{i} = x((n_vol+1):end);
    losses_M2(i) = xcost;
end


% stage two 
[~,i] = min(losses_M2);
a_lms_est_M2 = a_lms_est{i};
b_pu_est_M2 = b_pu_est{i};
x_initial = [a_lms_est_M2; b_pu_est_M2];

fun = @(x) find_cost_grad_a_b(x, m1_est, m2_est, m3_est,...
    l1, l2, l3, Phi_lms_nodes, Psi, SO3_rules); 
[x, loss_M3] = fmincon(fun,x_initial,A,b,[],[],[],[],[],options);

a_lms_est_M3 = x(1:n_vol);
b_pu_est_M3 = x((n_vol+1):end);

t2 = toc; 
%% align and save the reconstructed volume 
[~, a_lms_est_M2, reflect2] = vol_coeffs_align(a_lms_est_M2, a_lms, L, S, 1e4);
[~, a_lms_est_M3, reflect3] = vol_coeffs_align(a_lms_est_M3, a_lms, L, S, 1e4);

V_est_M2 = vol_coeffs_t_vol(a_lms_est_M2, n, c, L, S);
V_est_M3 = vol_coeffs_t_vol(a_lms_est_M3, n, c, L, S);


WriteMRC(V_gt, 3*1.117, 'recons/Vol_ground_truth.mrc')
WriteMRC(V_est_M2, 3*1.117, 'recons/Vol_recon_first_two_MoMs.mrc')
WriteMRC(V_est_M3, 3*1.117, 'recons/Vol_recon_first_three_MoMs.mrc')

disp(['running time for reconstruction is ', ...
    num2str(t2-t1)])

disp(['the relative l2 error is ', ...
    num2str(norm(V_est_M3(:)-V_gt(:))/norm(V_gt(:)))])


