##### What is MDR (Multi Dimentionality Reduction)?
* Some of the popular dimentionality reduction techinques : https://towardsdatascience.com/dimensionality-reduction-toolbox-in-python-9a18995927cd
    * PCA
    * Kernel PCA 
    * Incremental PCA
    * Sparse PCA
    * SVD
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
