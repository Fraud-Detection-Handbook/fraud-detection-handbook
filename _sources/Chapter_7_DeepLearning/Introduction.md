(Deep_learning)=
# Introduction

The most widely used models for fraud detection in the industry or in machine learning competitions {cite}`kaggle2019fraud` are gradient boosting algorithms such as XGBoost {cite}`chen2016xgboost`, LightGBM {cite}`ke2017lightgbm`, CatBoost {cite}`prokhorenkova2017catboost`, and tree-based models such as random forest {cite}`breiman2001random`. With the right preprocessing and feature engineering, these models provide very convincing results in real-world fraud detection systems.

Neural network algorithms are less often considered in fraud benchmarks with static data as they are more difficult to tune to reach a competitive predictive performance. However, they have many advantages which make them essential in a fraud detection practitioner's toolbox.

## Why using a neural network for fraud detection?

There is no reason to assume that a multi-layer feed-forward neural network could outperform random forest or XGBoost on static datasets but there are several other important criteria in the fraud detection problem in addition to the detection performance.

### Incremental learning

XGBoost and random forest are both tree ensembles. Decision trees are generally not incremental because they require the overall dataset to compute optimal splits and build their structure. Modifying a split given a novel set of data is far from trivial. In particular, as it is built hierarchically, updating a condition in a high split of a tree directly makes the subtrees' structure unusable. It is worth noting that a number of techniques have been proposed to incrementally update trees, like Hoeffding trees {cite}`domingos2000mining`, Mondrian trees {cite}`lakshminarayanan2014mondrian`, or incremental ensembles of trees {cite}`sun2018concept`. Tree-based algorithms however mostly remain used in a batch learning scenario.

Incremental learning is useful for fraud detection because (1) it is less resource-intensive as models can be updated often on the last chunks of data instead of having to be fully trained on the whole dataset from scratch every time, and (2) it discards the need to store historical data for a long time thus avoiding data regulation issues.

Neural networks have the advantage of being incremental by nature since their training is iterative and instance-wise.

### Representation learning and end-to-end training

Many studies have shown that additionally to raw transaction features, the use of expert feature engineering (building relevant aggregates based on the cardholder history of transactions) significantly improves the fraud detection rate {cite}`bahnsen2016feature,dal2014learned`. 

However, this process has limitations, primarily that of being dependent on expensive human expert knowledge. There have been attempts to replace manual aggregation through automatic learning of representations {cite}`fu2016credit,jurgovsky2018sequence,dastidar2020nag`. These methods are mainly based on neural networks (Autoencoders, convolutional neural networks, long short-term memory networks).

Moreover, on top of these learned representations, using a feed-forward neural network instead of XGBoost or random forests is more interesting as it allows training the whole model (representation part + classification part) from one end to the other. 

### Federated learning

Federated learning consists in sharing and training a model on multiple devices with each device keeping its data locally. The idea is to share an initial model between the devices, update it locally, and frequently federate the updates from all devices into a global model for everyone. In general, the global update is computed with methods like federated averaging {cite}`konevcny2016federated`, i.e. through a weighted average of each local model's weights. 

Contrary to tree-based models, neural networks with the same architecture can have their weights averaged, which makes them the first choice when it comes to federated learning.

### An additional model for stacking

Although neural networks might reach a global performance close to XGBoost or random forests, this does not mean that these different models catch the same fraud patterns. In particular, experiments often show that combining a tree-based approach and a neural network into a simple averaging ensemble can lead, thanks to diversity, to a better performance overall.

### Take-away message

Apart from detection performance, neural networks have several advantages for the credit fraud detection problem: they can be stacked to other models, they can be trained incrementally, they can easily be federated, they allow representation learning, and they can learn representations and classification together with end-to-end training.

## Content of the chapter

This chapter covers techniques to build neural networks for the fraud detection problem. Section 2 describes general considerations to design a first model (fully connected feed-forward neural network). The next sections explore more advanced deep learning techniques to learn useful representations from data. Section 3 and 4 respectively describe the use of autoencoders and sequential models (Convolutional neural networks, long short-term memory networks, and attention mechanism). Finally, section 5 describes the results of all the methods on real-world data, for comparison with the batch methods from chapter 5. 
