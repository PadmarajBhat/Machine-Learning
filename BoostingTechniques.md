##### Notes on different boosting techniques

* All boosting technique, in the core, focus on the enhancing on the weaker model in the subsequent models.
      * https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db: Nice read on the differences between 3 boosting algos
         * comparison here is mainly on behavior to categorical features.
               * sample data for example is also rich in category values
               * in the end catboost is rated at the top for its speed and accuracy, followed by xgboost and lastly the lightgbm

* AdaBoost: Adaptive Boost (oldest among others)
    * The data points are initially given equal weight
    * when a weak tree/model does predictions on the data points
    * the weights are increased for those data points which are not nearer to the ground truth.
    * this enables the next model to correctly predict for the ones which currently have hiegher weight.
    * Now, the model **together** can predict outcome with better accuracy(or precision)
    * This continues till the number of trees are exhausted or
    * **early stopping** : defines the threshold of accuracy at which booting can be stopped.
    

* Gradient Boosting: 
   * Instead of increasing the weights to data points, it attempts to decrease the loss function
   * first a weak model predicts the output
   * residual error is calculated
   * seconds model uses the same input data but this tries to predict the residual error.
   * now, residual error is again calucated but 
         * first 2 predictions from 2 models are added then 
         * subracted from the ground truth
   * this new residual error is what third model tries to fit and predict
   * this continues until residual error goes to zero or number trees are exhausted
   * overfitting can be reduced by keeping track of validation score.

* eXtreme Gradient Boosting:
   * Before gettting into details of how it works, there are some nice usecases of XGBoost
         * using xgboost for image classification
         * xgboost as the replacement to dense + softmax in the neural network
         
   * xgboost supports 
         * distributed computing (for the cluster of machines) 
         * parallel computing ( cpu / gpc cores) 
         * out of core computing:  when data is outside the machine capacity.
         * cache optimized
         * continue training: https://stackoverflow.com/a/47000386/8693106
            ```bst = xgb.train(param0, dtrain2, num_round, evals=[(dtrain, "training")], xgb_model='xgbmodel')```
            
         * Learning:
                    * optimimum number of split is decided based on the gain. 
                    * if gain decreases to negative value then stop it or
                    * split till the max depth though there is negative gain in anticipation of positive gain in the deeper level.
                    * a greedy approach is taken to best split by adding gains from the branch.
         * How does it handle missing value?
         * Ho to do interactive feature analysis?
      
   
