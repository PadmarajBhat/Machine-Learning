##### What is Shap and its alternative approaches?
    * Quick intro : https://www.kaggle.com/dansbecker/shap-values

* Nice article on SHAP: https://towardsdatascience.com/how-to-avoid-the-machine-learning-blackbox-with-shap-da567fc64a8b
      * fitting a simpler model like linear regression on the **model** prediction to make following explanation
            * Prediction explainer: median prediction and attributes moving the prediction value up or down in case of regression model.
            * Model Explainer: Order of similarity based plot of all the feature to see impact of feature values to prediction.
            * Dependecy Plot:  SHAP values indicating the importance of features to the dependent feature.
            * Summary Plot: SHAP values indicating the extreme values impact on prediction for each features.

* python tool : https://github.com/slundberg/shap
      * shap.TreeExplainer :  for Boosted Tree explainability
            * shap.forceplot : for predicion and model explainer
            * shap.dependence_plot 
            * shap.summary_plot
