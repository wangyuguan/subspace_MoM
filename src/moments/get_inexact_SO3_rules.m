function SO3_rules_MoMs = get_inexact_SO3_rules(L,P,tau1,tau2,tau3)
Q1 = size(get_SO3_rule(L,P,1,0),1);
Q2 = size(get_SO3_rule(L,P,2,0),1);
Q3 = size(get_SO3_rule(L,P,3,0),1);

i = 0;
while true 
    if size(get_SO3_rule(L,P,1,i))<=(Q1*(1-tau1))
        break
    end
    i=i+1;
end
SO3_rules_MoMs.m1 = get_SO3_rule(L,P,1,i);


i = 0;
while true 
    if size(get_SO3_rule(L,P,2,i))<=(Q2*(1-tau2))
        break
    end
    i=i+1;
end
SO3_rules_MoMs.m2 = get_SO3_rule(L,P,2,i);



i = 0;
while true 
    if size(get_SO3_rule(L,P,3,i))<=(Q3*(1-tau3))
        break
    end
    i=i+1;
end
SO3_rules_MoMs.m3 = get_SO3_rule(L,P,3,i);

end