function [res1, res2] = get_fsc(V,V1,V2,pixelsize)

fsc1 = FSCorr(V,V1);
fsc2 = FSCorr(V,V2);
n = numel(fsc1);

plot(1:n,fsc1,'r-*','LineWidth',2); % Plot FSC

hold on;
plot(1:n,fsc2,'b-*','LineWidth',2); % Plot FSC
hold off;

xlim([1 n]);
ylim([-0.1 1.05]);
grid on

y=ones(1,n)*.5;
hold on; 
plot(1:n,y,'k--','Linewidth',1.5);
hold off;

j1 = fscres(fsc1,.5);
j2 = fscres(fsc2,.5);


res1=2*pixelsize*n/j1;
res2=2*pixelsize*n/j2;

xticks=get(gca,'XTick');
df=1/(2*pixelsize*n);
xticks=xticks*df;
set(gca,'XTickLabel',sprintf('%7.3f\n',xticks));
fontsize(gcf,16,"points");
xlabel('$1/\textup{\AA}$', 'Interpreter', 'latex', 'FontSize', 20)
set(gca,'FontSize',20)
legend({strcat([num2str(res1,'%.2f'),'$\textup{\AA}$']), strcat([num2str(res2,'%.2f'), '$\textup{\AA}$'])},'Interpreter','latex','FontSize',20)



end