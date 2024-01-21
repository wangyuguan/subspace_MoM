function jn = sphbes(n, z, derivative)

% if derivative == false
%     jn = sqrt(pi./(2*z)).*besselj(n+.5,z);
% else
%     jn = sqrt(pi./(2*z)).*besselj(n+.5,z);
%     j1 = sqrt(pi./(2*z)).*besselj(n-1+.5,z);
%     jn = j1 - jn*(n+1)./z;
% end


% if n==0
%     jn(z==0) = 1;
% else
%     jn(z==0) = 0;
% end

if derivative == false
    jn = sqrt(pi./(2*z)).*besselj(n+.5,z);
    if n==0
        jn(z==0)=1;
    else
        jn(z==0)=0;
    end
else
    jn = sqrt(pi./(2*z)).*besselj(n+.5,z);
    if n==0
        j1=sqrt(pi./(2*z)).*besselj(1+.5,z);
        jn=-j1+n*jn./z;
        jn(z==0)=0;
    end
    if n==1
        j1=sqrt(pi./(2*z)).*besselj(.5,z);
        jn=j1-(n+1)*jn./z;
        jn(z==0)=1;
    end
    if n>1
        j1=sqrt(pi./(2*z)).*besselj(n-1+.5,z);
        jn=j1-(n+1)*jn./z;
        jn(z==0)=0;
    end

end



end




