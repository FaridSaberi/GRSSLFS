function B = VBE(X,r)
%------------ Constructing the Basis for Feature Space via the Variance information and the Basis Extension (VBE) method

%------------ Inputs
%% X: Data matrix
%% r: rank of X

%------------ Output
%%  B: Basis matrix

E = var(X);
[~,index] = sort(E,'descend');
A = index;
B(:,1) = X(:,A(1));
j = 2;
for i = 2:r
B(:,i) = X(:,A(j));
while rank(B)~ = i
j = j+1;
B(:,i) = X(:,A(j));
end
j = j+1;
end
end 
