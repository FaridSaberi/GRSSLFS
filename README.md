# A Self-Representation Learning Method for Unsupervised Feature Selection using Feature Space Basis

Current methods of feature selection based on a self-representation framework use all the features of the original data in their representation framework. This issue carries over redundant and noisy features into the representation space, thereby diminishing the quality and effectiveness of the results. This work proposes a novel representation learning method, dubbed GRSSLFS (Graph Regularized Self-Representation and Sparse Subspace Learning), that mitigates the drawbacks of using all features. GRSSLFS employs an approach for constructing a basis for the feature space, which includes those features with the highest variance. The objective function of GRSSLFS is then developed based on a self-representation framework that combines subspace learning and matrix factorization of the basis matrix. Moreover, these basis features are incorporated into a manifold learning term to preserve the geometrical structure of the underlying data.
We provide an effectiveness and performance evaluation on several widely-used benchmark datasets. The results show that GRSSLFS achieves a high level of performance compared to several classic and state-of-the-art  feature selection methods.



This repository provides an implementation for the GRSSLFS method (Graph Regularized Self-Representation and Sparse Subspace Learning) as described in the paper:

Prayag Tiwari, Farid Saberi-Movahed, Saeed Karami, Farshad Saberi-Movahed, Jens Lehmann, Sahar Vahdati, A Self-Representation Learning Method for Unsupervised Feature Selection using Feature Space Basis, Under Review, 2024.

For any inquiries, please feel free to reach out to us via email: fdsaberi@gmail.com
