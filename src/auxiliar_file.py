# ============================================= list of models to use ============================================= 
# ["RandomForestClassifier","GradientBoostingClassifier","AdaBostClassifier","DecisionTree","LogisticRegression","ElasticNet","SVC","XGBoost","GaussianDB","LGBM","MLPClassifier"]


# =============================================  cross_validation ============================================= 
# use the cross validation to get the average metrics of the model
        # cross_results = cross_validate(model, x_train, y_train, scoring=['accuracy', 'balanced_accuracy', 'f1_weighted', 'matthews_corrcoef', 'roc_auc', 'roc_auc_ovr_weighted'])
        # results[model.__class__.__name__] = {
        #     'accuracy' : np.mean(cross_results['test_accuracy']),
        #     'balanced_accuracy' : np.mean(cross_results['test_balanced_accuracy']),
        #     'f1_weighted' : np.mean(cross_results['test_f1_weighted']),
        #     'matthews_corrcoef' : np.mean(cross_results['test_matthews_corrcoef']),
        #     'roc_auc_ovr_weighted' : np.mean(cross_results['test_roc_auc_ovr_weighted']),
        # }


# ============================================= prints ============================================= 
# print("Shape before: ",X.shape,"\n")
# print("Shape after: ",X_transformed.shape,"\n")
# print("Info of the dataset after the preprocessing: ", pd.DataFrame(X_transformed).info(),"\n")


"""
============================================= config.json ============================================= 
{
    "dataset" : "./dataset/train_students.csv",
    "norm_model" : "Standard",
    "number_best_features" : 30,
    "variance_threshold" : 0,
    "models_names": {
        "LogisticRegression" : {
            "penalty" : [null, "l1", "l2", "elasticnet"],
            "solver" : ["saga", "sag", "lbfgs", "liblinear"],
            "warm_start" : [true, false],
            "max_iter" : [100, 300, 600],
            "multi_class" : ["auto", "over", "multinomial"]
        }
    }
}
"""