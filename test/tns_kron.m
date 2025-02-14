function C = tns_kron(A, B)  % version 5
C = squeeze(reshape(A(:) * B(:).', [size(A), size(B)]));
end