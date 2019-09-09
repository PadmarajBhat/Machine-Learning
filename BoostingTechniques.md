##### Notes on different boosting techniques

* All boosting technique, in the core, focus on the enhancing on the weeker model in the subsequent models.

* AdaBoost: Adaptive Boost (oldest among others)
    * The data points are initially given equal weight
    * when a week tree/model does predictions on the data points
    * the weights are increased for those data points which are not nearer to the ground truth.
    * this enables the next model to correctly predict for the ones which currently have hiegher weight.
    * Now, the model **together** can predict outcome with better accuracy(or precision)
    * This continues till the number of trees are exhausted or
    * **early stopping** : defines the threshold of accuracy at which booting can be stopped.
    
