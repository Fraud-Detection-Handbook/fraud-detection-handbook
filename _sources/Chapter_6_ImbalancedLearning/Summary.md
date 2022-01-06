# Summary

This chapter covered the use of imbalanced learning techniques in the context of credit card fraud detection. A wide range of approaches were considered, spanning cost-sensitive, resampling, and ensemble techniques. For each approach, the experimental evaluation included a toy example, a dataset of simulated transactions, and a real-world dataset.

One of the main take-away of the experiments carried out in this chapter is that the benefits of imbalanced learning techniques are mitigated. In most cases, they allow to improve performance metrics such as AUC ROC, balanced accuracy, and training times. At the same time, they are usually detrimental to metrics like Average Precision and CP@100. 

It is worth noting that most of the literature on imbalanced learning techniques for credit card fraud detection relies on AUC ROC, balanced accuracy, and training times for motivating their use. Depending on the metric to be optimized, one may therefore find some of these results or conclusions misleading (see also {cite}`makki2019experimental`): as discussed in the [performance metrics summary](Summary_Performance_Metrics), the Average Precision and CP@100 should also be included in the optimization of a fraud detection system. 

Overall, the best prediction performances were obtained with [ensemble methods](Ensembling_Strategies). Imbalanced learning techniques allowed to provide slight improvements in terms of AUC ROC or training times for balanced bagging and balanced random forest. XGBoost appeared as the best performing model in most of the experiments, illustrating its robustness to data imbalance scenarios across all performance metrics. The most likely explanation is that the residuals naturally give more weight to the minority class, thus acting like a cost-sensitive technique.

