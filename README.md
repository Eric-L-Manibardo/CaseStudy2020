# Case study 2020 of DDD

The following figure describes the proposed experimental setup. For each traffic data source, ten points of the network are selected, always trying to choose different locations that offer uneven traffic profiles. Then the regression dataset for each target placement is built, containing the recordings of one year. The first three weeks of every month are used for model training, while the remaining days are kept for testing. Now, in order to find the best hyperparameters for each learning method, three-fold cross-validation is performed, by using two weeks of every month as learning data and the remaining week of reserved training data as validation data. The mean of these three validation scores is used as a performance indicator by the Hyperopt library, which provides algorithms and parallelization tools for searching the best hyperparameter configuration in a most efficient way than traditional grid search. After thirty evaluations of each learning method, a traffic forecasting model is developed, this time, by using all available training data and the best hyperparameters configuration. Finally, model performance metrics are computed upon data held for the test. Implementation code and simulations are available at this Github repository.

<p align="center">
<img src="https://github.com/erasperiko/CaseStudy2020/blob/master/Experimental%20Set%20Up.png" width="700" height="800" ></img>
<p>

| Location         | Data nature |  Scope  | Sensor type | Time resolution | Year |                              Data source                              |
|------------------|:-----------:|:-------:|:-----------:|:---------------:|:----:|:---------------------------------------------------------------------:|
| Madrid city      |     Flow    |  Urban  |   Roadside  |        15       | 2018 | https://datos.madrid.es/portal/site/egob/                             |
| California state |     Flow    | Freeway |   Roadside  |        5        | 2017 | http://pems.dot.ca.gov/                                               |
| New York city    |    Speed    |  Urban  |   Roadside  |        5        | 2016 | https://www.kaggle.com/crailtap/nyc-real-time-traffic-speed-data-feed |
| Seattle city     |    Speed    | Freeway |   Roadside  |        5        | 2015 | https://github.com/zhiyongc/Seattle-Loop-Data                         |


---

---
#### If you use any dataset in your work, please cite the following reference:
###### Reference:

###### BibTex:
```

```
#### Note: These datasets should only be used for research.
