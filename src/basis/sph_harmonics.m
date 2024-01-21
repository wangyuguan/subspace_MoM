function Yl = sph_harmonics(l, th, phi)

% Pl = legendre(l,cos(th));
% 
% m = 1:l;
% Pl = [flip(diag(((-1).^m) .* (factorial(l-m) ./ factorial(l+m))) * Pl(2:end,:),1); Pl];
% 
% m = (-l:l).';
% Yl = diag(sqrt(((2*l+1)/(4*pi))*factorial(l-m)./factorial(l+m))) * (Pl.* exp(1j*m*phi'));
% Yl = Yl.';
Yl = [];
for m=-l:l
    Yl = [Yl harmonicY(l,m,th,phi)];
end

end