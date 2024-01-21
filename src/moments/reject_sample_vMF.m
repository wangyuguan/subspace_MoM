function Rots = reject_sample_vMF(N,mus,w,k,M)

Rots = zeros(3,3,N);

parfor i=1:N
    Rots(:,:,i) = sample_rot(mus,w,k,M);
end

end


function rot = sample_rot(mus,w,k,M)


while true
    alpha = rand*2*pi;
    gamma = rand*2*pi;
    beta = acos(1-2*rand);

    x = [sin(beta)*cos(alpha);sin(beta)*sin(alpha);cos(beta)];
    f = vonMisesFisher3D(mus,w,x,k);

    if rand<(f/M)
        break
    end
    
end

rot = Rz(alpha)*Ry(beta)*Rz(gamma); 

end


% function Rots = reject_sample_rots(N, b_pu, P, M)
% 
% Rots = zeros(3,3,N);
% count = 0;
% 
% 
% [real_t_complex,~] = get_view_real_t_complex(P);
% B_pu = real_t_complex*[1; b_pu];
% n_B = numel(B_pu);
% vec_idx = @(p, u) p^2+u+p+1; 
% 
% while count<N
% 
% 
%     alpha = rand*2*pi;
%     gamma = rand*2*pi;
%     beta = acos(1-2*rand);
%     Rot = Rz(alpha)*Ry(beta)*Rz(gamma); 
% 
%     % Rot = randrot(3);
%     % [alpha,beta,gamma] = rot2elr(Rot);
% 
%     Psi = zeros(n_B,1);
%     for p=0:P
%         Wig_p = wignerD(p, alpha, beta, gamma); 
%     
%     
%         for u=-p:p
%             Psi(vec_idx(p,u)) = Wig_p(u+p+1,0+p+1);
%         end
%     end
%     f = real(sum(Psi.*B_pu));
%     % g = sin(beta)/8/pi^2;
%     g = 1;
%     
%     if rand<(f/(g*M))
%         count=count+1;
%         Rots(:,:,count) = Rot;
%         count
%     end
% 
%     
% 
% end
% 
% 
% end