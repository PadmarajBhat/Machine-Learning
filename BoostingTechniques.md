##### Notes on different boosting techniques

* All boosting technique, in the core, focus on the enhancing on the weaker model in the subsequent models.

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
