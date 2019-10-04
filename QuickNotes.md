##### Notes on Machine Learning from various Sources:
* R2 vs Adjusted R2 : https://discuss.analyticsvidhya.com/t/difference-between-r-square-and-adjusted-r-square/264
    * R2 increases if the variables are added
    * adjusted R2 would decrease or remain same if the newly added variable do not impact dependent variable.
    * highly advised to use adjusted R2 when multivariable linear regression is opted.
    
* Accelerating sklearn training through sk-dist : https://medium.com/building-ibotta/train-sklearn-100x-faster-bec530fc1f45

* interesting article on loss function: https://medium.com/analytics-vidhya/a-detailed-guide-to-7-loss-functions-for-machine-learning-algorithms-26e11b6e700b
   * loss function or error function: defines the error (difference of prediction and ground truth)
      * regression loss functions:
         * why do we need below functions? why cant we take the ypred-y as the loss function?
            * it is not fool proof. there may be chances that average calculation may result in 0 (due to possiblities of negative and positive preditions)
         * l2 loss - square of the difference between ypred and y
         * l1 loss - mod of the difference between ypred and y
         * huber loss - half of squre of difference if the value is less than delta 
                        if not less then mod of the difference
      * classification loss functions:
         * Binary cross entropy loss, multiclass cross entropy loss
         * Hinge loss
         * KL - Divergence
   * Cost Function: is the average loss for all the training data

* https://www.datasciencecentral.com/profiles/blogs/google-releases-massive-visual-databases-for-machine-learning
* https://ml.dask.org/joblib.html
* https://twitter.com/dask_dev/status/1175049472520310788?s=19
* Time series forecast: https://www.datasciencecentral.com/video/dsc-webinar-series-how-to-use-time-series-data-to-forecast-at
* ML guide: https://www.datasciencecentral.com/profiles/blogs/supervised-learning-everything-you-need-to-know
* Ml algo : quick read
