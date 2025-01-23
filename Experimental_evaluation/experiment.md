# **Experimental evaluation**



---

<div style="text-align: justify;">


In this study, an extensive experimental evaluation was conducted, involving the feature descriptors presented in Tables 2, 3, and 4, as well as all the algorithms listed in Table 1. To define this phase, the experiments were organised according to the type of information to be processed by the machine learning algorithms. In this context, the CNN algorithm was evaluated using the Mol2Vec, Fasta2Seq, and Smiles2Seq descriptors, which facilitate the retention of dependency information between the attributes. The use of the CNN algorithm in conjunction with these descriptors aims to explore the existence of sequential/spatial relationships among the input variables. In contrast, traditional machine learning algorithms treat the input variables as independent from each other. This means that, during the model induction process, no explicit consideration is given to positional/spatial dependency relationships between the various input variables. In this regard, all traditional machine learning algorithms were evaluated with each of the descriptors.

To determine the optimal parameters for each of the classification algorithms used, we employed the Bayesian Hyperparameter Optimisation approach. The concept behind this method is to find the maximum value of an unknown function, in this case, the performance of a machine learning algorithm, with a minimal number of iterations. Initially, a hypothesis about the function is formed, and then this hypothesis is continuously updated to form a posterior distribution that integrates the information acquired from the observed data. At each iteration, the function is optimised to select the next query point, aiming to achieve the greatest possible improvement compared to the best observation so far. This iterative process is repeated until a stopping criterion is met, which could be a pre-determined number of iterations or when the improvements become negligible. In this study, we implemented this approach using the BayesSearchCV function from the scikit-optimize library, with a total of 10 iterations and accuracy as the performance metric.

In this study, we employed the K-fold Cross-Validation (CV) technique. This analytical method involves splitting the data into k partitions, where the data from k-1 partitions are used to train a predictor, and subsequently, the model is tested on the remaining partition. This process is repeated k times, with each cycle using a different partition, to generate an average performance outcome. However, there is a possibility that the model may use the same data for parameter tuning in performance evaluation, potentially leading to a biased estimate of the model’s predictive performance. To address this issue, we employed the Nested Cross-Validation technique. This method applies double cross-validation. Consequently, a cross-validation process with cv_outer partitions (outer loop) is used to split the data into training and testing sets. Then, on the training set, a second cross-validation process with cv_inner partitions (inner loop) is carried out to fine-tune the model’s parameters. After this, using the best estimated parameters, we constructed and evaluated the model’s performance based on the test set data. The Stratification technique was employed in both processes to ensure that each partition had the same class distribution as the original dataset.

In this study, we used the parameters cv_inner=3 and cv_outer=2 for the Nested Cross-Validation strategy. This cross-validation method was repeated 5 times, yielding a total of 10 performance estimates for each evaluation performed. To assess the classification models in this study, we used several metrics, including accuracy (Acc), precision (prec), recall (rec), F1 score (f1), and Matthews correlation coefficient (mcc). The models were built for binary classification, considering the anticancer and non-anticancer classes. These metrics are calculated based on the parameters: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). TP represents instances of the anticancer class correctly classified, TN represents instances of the non-anticancer class correctly classified, FP represents instances classified as anticancer but actually belonging to the non-anticancer class, and FN represents instances classified as non-anticancer but actually belonging to the anticancer class. The total number of instances is given by n = TP + TN + FP + FN. The definition of each metric used in this study can be found in Table 5.

**Table 5** \- Performance Metrics

| Metric | Definition |
| :---: | :---: |
| **Accuracy (Acc)** | **acc=(TP+TN)/n** |
| **Matthews correlation coefficient (mcc)** | **mcc=(TPxTN)-(FPxFN)/((TP+FP)x(TP+FN)x(TN+FP)x(TP+FN))1/2** |
| **F1-score (f1)** | **f1=(2 x prec x rec)/(prec+rec)** |
| **Precision (prec)** | **prec=TP/(TP+FP)** |
| **Recall (rec)** | **rec=TP/(TP+FN)** |
</div>
The author (2023)
