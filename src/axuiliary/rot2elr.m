function [alpha, beta, gamma] = rot2elr(Rot)

if Rot(3,3)<1

    if Rot(3,3)>-1

        beta = acos(Rot(3,3));
        alpha = atan2(Rot(2,3),Rot(1,3));   
        gamma = atan2(Rot(3,2),-Rot(3,1));

    else
    
        beta = pi;
        alpha = -atan2(Rot(2,1),Rot(2,2));    
        gamma = 0;

    end

else

    beta = 0;
    alpha = atan2(Rot(2,1),Rot(2,2));
    gamma = 0;


end


if alpha<0
    alpha = alpha+2*pi;
end

if gamma<0
    gamma = gamma+2*pi;
end


end

