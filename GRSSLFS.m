function Selected_feat = GRSSLFS(X,BB,BX,P,A,alpha,beta,gamma,k,itermax)
%------------ The proposed GRSSLFS method:

%------------ Inputs
%% X: Data matrix in R^(m*n), withe m samples and n features
%% B: Basis for the feature space
%% A: Similarity matrix in R^(n*n) associated with the features
%% P: Degree matrix
%% alpha,beta,gamma: Regularization parameters
%% k: The number of selected features
%% itermax: Maximum number of iterations
%% BB = (B')*B;
%% BX = (B')*X;
%------------ Output
%% Selected_feat: Selected features

%------------ Initializations
[m,n] = size(X);
G = rand(m,n);
U = rand(n,k);
V = rand(k,m);
onesm = ones(m,m);

%------------ Calculation of the matrix E used in the L2,1 norm
a = 4*max(diag(U*(U')),10^-9);
E = diag(sqrt(1./a));

for i  =  1:itermax
%------------ Update G
VU = (V')*(U');
BBG = BB*G;
NoG = BX+alpha*BBG*A+BB*VU;
DeG = BBG+alpha*BBG*P+BBG*(VU')*VU;
re1 = rdivide(NoG,DeG);
G = G.*nthroot(re1,2);

%------------ Update U
GBB = (G')*BB;
NoU = GBB*(V');
DeU = GBB*G*U*V*(V')+beta*E*U;
re2 = rdivide(NoU,DeU);
U = U.*nthroot(re2,2);

%------------ Update V
UGB = BB*G*U;
NoV = (UGB')+gamma*V;
DeV = (UGB')*G*U*V+gamma*V*onesm;
re3 = rdivide(NoV,DeV);
V = V.*nthroot(re3,2);

%------------ Update E
a = 4*max(diag(U*(U')),10^-9);
E  =  diag(sqrt(1./a));
end

%------------ Feature Selection Process
tempVector  =  sum(U.^2, 2);
[~, value]  =  sort(tempVector,'descend'); 
Selected_feat  =  value(1:k);
end
