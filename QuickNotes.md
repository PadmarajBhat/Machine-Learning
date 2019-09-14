##### Notes on Machine Learning from various Sources:
* R2 vs Adjusted R2 : https://discuss.analyticsvidhya.com/t/difference-between-r-square-and-adjusted-r-square/264
    * R2 increases if the variables are added
    * adjusted R2 would decrease or remain same if the newly added variable do not impact dependent variable.
    * highly advised to use adjusted R2 when multivariable linear regression is opted.
    
* Accelerating sklearn training through sk-dist : https://medium.com/building-ibotta/train-sklearn-100x-faster-bec530fc1f45

* interesting article on loss function: https://medium.com/analytics-vidhya/a-detailed-guide-to-7-loss-functions-for-machine-learning-algorithms-26e11b6e700b
   * loss function or error function: defines the error (difference of prediction and ground truth)
   * Cost Function: is the average loss for all the training data
