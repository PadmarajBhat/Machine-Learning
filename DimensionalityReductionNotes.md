##### What is MDR (Multi Dimentionality Reduction)?
* Some of the popular dimentionality reduction techinques : https://towardsdatascience.com/dimensionality-reduction-toolbox-in-python-9a18995927cd
    * PCA
    * Kernel PCA 
    * Incremental PCA
    * Sparse PCA
    * SVD
      * can be used for : https://medium.com/analytics-vidhya/master-dimensionality-reduction-with-these-5-must-know-applications-of-singular-value-777299940b89
         * image compression:
            * the decomposing high matrix to a low rank matrix retain maximum information through svd
         * image recovery
            * masked images can be decomposed an dthe lower ranked matrices can be used to replace the original masked image.
         * svd for eigen faces
            * extracting most import features of the faces
         * spectral clustering
         * removing background from videos
      * ways to use svd in python
         * ```
            from numpy.linalg import svd ; svd(nd array)
            ```
         * ```
              from sklearn.decomposition import TruncatedSVD
              svd =  TruncatedSVD(n_components = 2)
              A_transf = svd.fit_transform(A)
           ```
           
         * ```
            from sklearn.utils.extmath import randomized_svd
            A = np.array([[-1, 2, 0], [2, 0, -2], [0, -2, 1]])
            u, s, vt = randomized_svd(A, n_components = 2)
            ```
    * Gaussian Random Projection
    * Sparse Random Projection
    * Multidimension Scaling
    * ISOMAP
    * Minibatch Dictionary Learning
    * Independent Component Analysis
    * T-distributed Stochastic Neighboring Embedding
    * Locally Linear Embedding
    * Autoencoder
    
    
 * There are also following iterative approaches for dimensionality reduction:
   * Backword feature reduction: iteratively feature variables or columns are reduced one by one to know the significance with dependent variable and independent variables
   * forward feature reduction: iteratively feature variables or columns are added one by one to know the significance with both dependent and independent variables.
         * In both the cases my intuition is that it is heavily human centric approach and time consuming. May be we can programatically iterate over the features (in either of the above cases) to find out the least average co variance between the independent variables and high variance with dependent variable. However, this is costly operation.
