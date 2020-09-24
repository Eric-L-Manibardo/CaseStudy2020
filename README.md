# Case study 2020 of DDD

> The following figure describes the proposed experimental setup. For each traffic data source, ten points of the network are selected, always trying to choose different locations that offer uneven traffic profiles. Then the regression dataset for each target placement is built, containing the recordings of one year. The first three weeks of every month are used for model training, while the remaining days are kept for testing. Now, in order to find the best hyperparameters for each learning method, three-fold cross-validation is performed, by using two weeks of every month as learning data and the remaining week of reserved training data as validation data. The mean of these three validation scores is used as a performance indicator by the Hyperopt library, which provides algorithms and parallelization tools for searching the best hyperparameter configuration in a most efficient way than traditional grid search. After thirty evaluations of each learning method, a traffic forecasting model is developed, this time, by using all available training data and the best hyperparameters configuration. Finally, model performance metrics are computed upon data held for the test. Implementation code and simulations are available at this Github repository.

> <img src="https://github.com/erasperiko/CaseStudy2020/blob/master/Experimental%20Set%20Up.png" ></img>



---

---
#### If you use any dataset in your work, please cite the following reference:
###### Reference:

###### BibTex:
```

```
#### Note: These datasets should only be used for research.
