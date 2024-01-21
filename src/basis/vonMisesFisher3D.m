function f = vonMisesFisher3D(mus,w,x,k)
n = size(mus,2);
f = 0;
for i=1:n
    mu = mus(:,i);
    f = f+w(i)*exp(k*sum(mu.*x))*k/2/pi/(exp(k)-exp(-k));
end
end