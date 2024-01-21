function Phi_lms_nodes = precompute_PCs(U, L, S, n, c, SO3_rule)

sphbes_zeros = zeros([L+1,max(S)]);
for l=0:L
    sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
end

if mod(n,2)==0
    k = ((-n/2):(n/2-1))/n;  
else
    k = ((-(n-1)/2):((n-1)/2))/n; 
end
[kx,ky] = meshgrid(k);
kx = kx(:); ky = ky(:); 


real_t_complex = get_vol_real_t_complex(L,S);


q = size(SO3_rule,1);
Phi_lms_nodes = cell(1,q);


parfor i=1:q
    Phi_lms_nodes{i} = zeros(size(U,2),size(real_t_complex,2));
    alpha = SO3_rule(i,1);
    beta = SO3_rule(i,2);
    gamma = SO3_rule(i,3);

    rotmat = elr2rot(alpha, beta, gamma);
    grids = rotmat'*[kx';ky';zeros(size(ky'))]; 
    grids = grids';
    [r,th,phi] = Cart2Sph(grids(:,1), grids(:,2), grids(:,3));

    

    idx = 1;

    for l=0:L
        Y_l = sph_harmonics(l, th, phi);
        zl = sphbes_zeros(l+1,:);
        cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));
        f_l = sphbes(l, r*zl/c, false)*diag(cl); 
        f_l(r>c) = 0;

        for s=1:S(l+1)
            f_ls = f_l(:,s);
            for m=-l:l
                Y_lm = Y_l(:,m+l+1);
                F_lms = Y_lm.*f_ls;

                Phi_lms_nodes{i}(:,idx) = U'*F_lms;
                idx = idx+1;
            end
        end
    end

    Phi_lms_nodes{i} = Phi_lms_nodes{i}*real_t_complex;
    
end

end


% function [PCs_tns, PCs_cell] = precompute_PCs(U, L, S, n, c, SO3_rule)
% 
% sphbes_zeros = zeros([L+1,max(S)]);
% for l=0:L
%     sphbes_zeros(l+1,:) = besselzero(l+.5,max(S),1);
% end
% 
% if mod(n,2)==0
%     k = ((-n/2):(n/2-1))/n;  
% else
%     k = ((-(n-1)/2):((n-1)/2))/n; 
% end
% [kx,ky] = meshgrid(k);
% kx = kx(:); ky = ky(:); 
% 
% real_t_complex = get_vol_real_t_complex(L,S);
% n_vol = size(real_t_complex,1);
% d = size(U,2);
% 
% q = size(SO3_rule,1);
% PCs_tns = zeros(q, d, n_vol);
% 
% 
% for i=1:q
% 
%     alpha = SO3_rule(i,1);
%     beta = SO3_rule(i,2);
%     gamma = SO3_rule(i,3);
% 
%     rotmat = elr2rot(alpha, beta, gamma);
%     grids = rotmat'*[kx';ky';zeros(size(ky'))]; 
%     grids = grids';
%     [r,th,phi] = Cart2Sph(grids(:,1), grids(:,2), grids(:,3));
% 
%     idx = 1;
% 
%     for l=0:L
%         Y_l = sph_harmonics(l, th, phi);
%         zl = sphbes_zeros(l+1,:);
%         cl = sqrt(2/c^3)./abs(sphbes(l, zl, true));
%         f_l = sphbes(l, r*zl/c, false)*diag(cl); 
%         f_l(r>c) = 0;
% 
%         for s=1:S(l+1)
%             f_ls = f_l(:,s);
%             for m=-l:l
%                 Y_lm = Y_l(:,m+l+1);
%                 F_lms = Y_lm.*f_ls;
% 
%                 PCs_tns(i,:,idx) = U'*F_lms;
%                 idx = idx+1;
%             end
%         end
%     end
% 
% end
% 
% PCs_tns = tns_mult(PCs_tns,3,real_t_complex,1);
% 
% 
% PCs_cell = cell(1,q);
% for i=1:q
%     PCs_cell{i} = squeeze(PCs_tns(i,:,:));
% end
% 
% 
% 
% end