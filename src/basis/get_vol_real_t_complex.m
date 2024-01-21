function [vol_real_t_complex, lin_idx] = get_vol_real_t_complex(L,S)

lin_idx = get_linear_idx(L,S);

vol_real_t_complex = [];
count = 1;

for l=0:L
    for s=1:S(l+1)
        for m=-l:l

            if m>0

                vol_real_t_complex(count, count) = 1;
                vol_real_t_complex(count, lin_idx(l+1,s,-m+l+1)) = -1j*((-1)^(l+m));

            elseif m==0

                vol_real_t_complex(count, count) = (1j)^l;

            else

                vol_real_t_complex(count,count) = 1j;
                vol_real_t_complex(count,lin_idx(l+1,s,-m+l+1)) = (-1)^(l+m);

            end
            
            count = count + 1;
        end
    end
end

vol_real_t_complex = sparse(vol_real_t_complex);

end


function lin_idx = get_linear_idx(L,S)

lin_idx = [];
count = 1;

for l=0:L
    for s=1:S(l+1)
        for m=-l:l
            lin_idx(l+1,s,m+l+1) = count; 
            count = count + 1;
        end
    end
end
        
end