# Deep Learning for Road Traffic Forecasting: Does it Make a Difference? 

# A Case Study 

The forecasting problem is formulated as a regression task, where the previous measurements of each target road collected at times {t-4, ... ,t} are used as features to predict the traffic measurement at the same location and time t+h. Four prediction horizons h={1,2,3,4} are considered, so that a separate single-step prediction model is trained for each h value and target location.

the Figure below describes the proposed experimental setup. For each traffic data source, 10 points of the road network are selected, always choosing locations that offer diverse traffic profiles. Then, a regression dataset for each target placement is built, covering data of one year. The first three weeks of every month are used for model training, whereas the remaining days are kept for testing. This split criterion allows verifying whether models are capable to learn traffic profiles that vary between seasons and vacations days. 

In order to find the best hyper-parameter values for each regression model, three-fold cross-validation is performed: two weeks of every month are used for training, and the remaining ones of the reserved training data are used for validation. The average of the three validation scores (one per every partition) is used as the objective function of a Bayesian optimizer, which searches for the best hyper-parameter configuration efficiently based on the aforementioned objective function. After evaluating 30 possible configurations for each model, the best hyper-parameter configuration is set on the model at hand, which is trained over all training data. Once trained, model performance scores are computed over the data held for testing. This process reduces the chances to have a bias in the comparisons later discussed due to a bad hyper-parameter configuration of the models.

All datasets, Python source code, details on the hyper-parameters sought for every model in the benchmark, sizes of Deep Learning models (number of trainable parameters), and simulation results are publicly available at this Github repository.

<p align="center">
<img src="https://github.com/Eric-L-Manibardo/CaseStudy2020/blob/master/experimental_setup.png" width="700" height="800" ></img>
<p>

# Dataset overview

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
