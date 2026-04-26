
%% Dense QP formulation
% Inputs:  A, B - systems dynamic
%          Q, R - cost function matrix
%             N - horizont
%            z0 - starting point
%     u_lb/u_ub - lower and upper bound for entries
%     z_lb/z_ub - lower and upper bound for states
% Output:
% The informations that give the following minimization problem 
%       min 1/2 x'H x + q'x
%          st: Cx <= d;%
%%
function[H,q,C,d]= denseMPC(A,B,Q,R,z0,N,u_ub, u_lb, z_ref, x_ref)
[nz,nu] = size(B);      % get the size of the state variable and entries
QQ = kron(eye(N),Q);
%QQ = blkdiag(QQ,P);     % Create the matrix with Q block on the diagonal
RR = kron(eye(N),R);    % Create the matrix with R block on the diagonal
bb = zeros(N*nz,nz);
bb(1:nz,:) = A; 
AA = kron(eye(N),B);    % Put B on diagonal
zz_ref = kron(ones(N,1),z_ref); 

xx_ref = kron(ones(N,1),x_ref);
%% Form the equal constrain: Ax = b
for i = 1: N-1
bb(i*nz +1: (i+1)*nz,:)= A^(i+1);
AA = AA + kron(diag(diag(ones(N-i)),-i), bb((i-1)*nz +1: i*nz,:)* B);
end
bb = bb *z0;
%% Form the inequality constrain: C x < d
Cu =kron(eye(N),[eye(nu); - eye(nu)]);
du = kron(ones(N,1),[u_ub; -u_lb]);

C = Cu;
d = du;
%% Find H and q
H = AA'*QQ* AA + RR;
q = AA' * QQ * bb -AA'*QQ*zz_ref - RR*xx_ref;
end