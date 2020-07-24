# ClassificationClassImbalance

## What is Class Imbalance?

An imbalanced classification problem is an example of a classification problem where the distribution of examples across the known classes is biased or skewed. The distribution can vary from a slight bias to a severe imbalance where there is one example in the minority class for hundreds, thousands, or millions of examples in the majority class or classes.

Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. Classification accuracy is often the first measure we use when evaluating models on our classification problems. Model can have excellent accuracy (such as 90%), but the accuracy is only reflecting the underlying class distribution. This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.

## Examples of Imbalanced Classification

Many industrial classification problems (real-world data intensive applications) in practice are imbalanced. The following examples have the nature of imbalanced classification:

- Fraud Detection.
- Claim Prediction
- Default Prediction.
- Churn Prediction.
- Spam Detection.
- Anomaly Detection.
- Outlier Detection.
- Intrusion Detection
- Conversion Prediction.
- Disease screening
- Advertising click-throughs

*****
*****

## Methods to handle imbalanced data

The first direct approach is to collect more data if possible. If the dataset can not be updated, then the imbalanced data problem can be addressed in **dataset level** or in the **algorithm level**.

----
### Data Level Approach: Resampling

Resampling techinques are used to either increasing the frequency of the minority class or decreasing the frequency of the majority class.

1. Random under-sampling

   Random Undersampling aims to balance class distribution by randomly eliminating majority class examples. This is done until the majority and minority class instances are balanced out.

   - **Advantages**

     help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.

   - **Disadvantages**

     - It can discard potentially useful information which could be important for building rule classifiers.

     - The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in inaccurate results with the actual test data set.

2. Random over-sampling

   Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.

   - **Advantages**

     - Unlike under sampling this method leads to no information loss.
     - Outperforms under sampling

   - **Disadvantages**
     - It increases the likelihood of overfitting since it replicates the minority class events.

3. Cluster-based over-sampling

   In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class have an equal number of instances and all classes have the same size. After oversampling of each cluster, all clusters of the same class contain the same number of observations.

   - **Advantages**

     - This clustering technique helps overcome the challenge between class imbalance. Where the number of examples representing positive class differs from the number of examples representing a negative class.
     - Also, overcome challenges within class imbalance, where a class is composed of different sub clusters. And each sub cluster does not contain the same number of examples.

   - **Disadvantages**
     - The main drawback of this algorithm, like most oversampling techniques is the possibility of over-fitting the training data.

4. Informed over-sampling: Synthetic Minority Over-sampling technique

   This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.

   ![image](Images\SMOTE.webp)

   - **Advantages**

     - Mitigates the problem of overfitting caused by random oversampling as synthetic examples are generated rather than replication of instances
     - No loss of useful information

   - **Disadvantages**

     - While generating synthetic examples SMOTE does not take into consideration neighboring examples from other classes. This can result in increase in overlapping of classes and can introduce additional noise

     - SMOTE is not very effective for high dimensional data

5. Modified synthetic minority oversampling (MSMOTE)
   It is a modified version of SMOTE. SMOTE does not consider the underlying distribution of the minority class and latent noises in the dataset. To improve the performance of SMOTE a modified method MSMOTE is used.

   This algorithm classifies the samples of minority classes into 3 distinct groups – Security/Safe samples, Border samples, and latent nose samples. This is done by calculating the distances among samples of the minority class and samples of the training data.

   Security samples are those data points which can improve the performance of a classifier. While on the other hand, noise are the data points which can reduce the performance of the classifier. The ones which are difficult to categorize into any of the two are classified as border samples.

   While the basic flow of MSOMTE is the same as that of SMOTE (discussed in the previous section). In MSMOTE the strategy of selecting nearest neighbors is different from SMOTE. The algorithm randomly selects a data point from the k nearest neighbors for the security sample, selects the nearest neighbor from the border samples and does nothing for latent noise.
----
### Algorithm Level Approach

To address the imbalanced classes challenge in classification problmes, modifying existing classification algorithms to make them appropriate for imbalanced data sets is an option when dataset can not be udpated. The main objective of **ensemble** methodology is to improve the performance of single classifiers. The approach involves constructing several two stage classifiers from the original data and then aggregate their predictions. Common ensemble methods are Bagging and Boosting to increase the performance of a single classifier.

#### 1.  Bagging based algorithm

    Bagging is an abbreviation of **Bootstrap Aggregating**. The conventional bagging algorithm involves generating ‘n’ different bootstrap training samples with replacement. And training the algorithm on each bootstrapped algorithm separately and then aggregating the predictions at the end. (use m subsets of samples, train n times)

    Bagging is used for reducing overfitting in order to create strong learners for generating accurate predictions. Unlike boosting, bagging allows replacement in the bootstrapped sample.

    ![image](Images\Bagging.webp)

    **Example of Bagging**

        Total Observations = 1000
        Fraudulent   Observations =20
        Non Fraudulent Observations = 980
        Event Rate= 2 %

        There are 10 bootstrapped samples chosen from the population with replacement. Each sample contains 200 observations. And each sample is different from the original dataset but resembles the dataset in distribution & variability.

        The machine learning algorithms like logistic regression, neural networks, decision tree  are fitted to each bootstrapped sample of 200 observations. And the Classifiers c1, c2…c10 are aggregated to produce a compound classifier.  This ensemble methodology produces a stronger compound classifier since it combines the results of individual classifiers to come up with an improved one.

    - **Advantages**
      - Improves stability & accuracy of machine learning algorithms
      - Reduces variance
      - Overcomes overfitting
      - Improved misclassification rate of the bagged classifier
      - In noisy data environments bagging outperforms boosting


    - **Disadvantages**
        - Bagging works only if the base classifiers are not bad to begin with Bagging bad classifiers can further degrade performance

#### 2. Boosting based algorithm

Boosting is an ensemble technique to combine weak learners to create a strong learner that can make accurate predictions. Boosting starts out with a base classifier / weak classifier that is prepared on the training data.

**What are base learners / weak classifiers?**

The base learners / Classifiers are weak learners i.e. the prediction accuracy is only slightly better than average. A classifier learning algorithm is said to be weak when small changes in data induce big changes in the classification model.

In the next iteration, the new classifier focuses on or **places more weight to those cases which were incorrectly classified in the last round**.

1) Adaptive Boosting: Ada Boosting techniques

    Ada Boost is the first original boosting technique which creates a highly accurate prediction rule by combining many weak and inaccurate rules.  **Each classifier is serially trained with the goal of correctly classifying examples in every round that were incorrectly classified in the previous round**.

    For a learned classifier to make strong predictions it should follow the following three conditions:

    - The rules should be simple
    - Classifier should have been trained on sufficient number of training examples
    - The classifier should have low training error for the training instances
        
    Each of the weak hypothesis has an accuracy slightly better than random guessing i.e. Error Term € (t) should be slightly more than ½-β where β >0. This is the fundamental assumption of this boosting algorithm which can produce a final hypothesis with a small error

    After each round, it gives more focus to examples that are harder to classify.  The quantity of focus is measured by a weight, which initially is equal for all instances. After each iteration, the weights of misclassified instances are increased and the weights of correctly classified instances are decreased.

    ![image](Images\Adaboosting.webp)

    **Example of Ada Boosting**

       1000 observations out of which 20 are labelled fraudulent. Equal weights W1 are assigned to all observations and the base classifier accurately classifies 400 observations.

       Weight of each of the 600 misclassified observations is increased to w2 and weight of each of the correctly classified observations is reduced to w3.

       In each iteration, these updated weighted observations are fed to the weak classifier to improve its performance. This process continues till the misclassification rate significantly decreases thereby resulting in a strong classifier.

    - **Advantages**
      - Very Simple to implement
      - Good generalization- suited for any kind of classification problem 
      - Not prone to overfitting


    - **Disadvantages**
        - Sensitive to noisy data and outliers

2) Gradient Tree Bossting

    In Gradient Boosting many models are trained **sequentially**. It is a **numerical optimization algorithm where each model minimizes the loss function**, y = ax+b+e, using the **Gradient Descent Method**.

    Decision Trees are used as weak learners in Gradient Boosting.

    While both Adaboost and Gradient Boosting work on weak learners / classifiers, try to boost them into a strong learner, there are some fundamental differences in the two methodologies. **Adaboost** either requires the users to specify a set of weak learners  or randomly generates the weak learners before the actual learning process. **The weight of each learner is adjusted at every step depending on whether it predicts a sample correctly**.

    On the other hand, **Gradient Boosting** builds the first learner on the training dataset to predict the samples, calculates the **loss (Difference between real value and output of the first learner)**. And use this loss to build an improved learner in the second stage.

    At every step, the residual of the loss function is calculated using the **Gradient Descent Method** and the new residual becomes a target variable for the subsequent iteration.

    ![image](Images\GradientBoosting.webp)

    **Example of Gradient Boosting**

       In a training data set containing 1000 observations out of which 20 are labelled fraudulent an initial base classifier. Target Variable Fraud =1 for fraudulent transactions and Fraud=0 for not fraud transactions.

       For eg: Decision tree is fitted which accurately classifying only 5 observations as Fraudulent observations. A differentiable loss function is calculated based on the difference between the actual output and the predicted output of this step.  The residual of the loss function is the target variable (F1) for the next iteration.

       Similarly, this algorithm internally calculates the loss function, updates the target at every stage and comes up with an improved classifier as compared to the initial classifier.

    - **Disadvantages**
       - Gradient Boosted trees are harder to fit than random forests
       - Gradient Boosting Algorithms generally have 3 parameters which can be fine-tuned, Shrinkage parameter, depth of the tree, the number of trees. Proper training of each of these parameters is needed for a good fit. If parameters are not tuned correctly it may result in over-fitting.

3) XG Boosting

    XGBoost (Extreme Gradient Boosting) is an advanced and more efficient implementation of Gradient Boosting Algorithm discussed in the previous section.

    - **Advantages**
       - It is 10 times faster than the normal Gradient Boosting as it implements parallel processing. It is highly flexible as users can define custom optimization objectives and evaluation criteria, has an inbuilt mechanism to handle missing values.
       - Unlike gradient boosting which stops splitting a node as soon as it encounters a negative loss, XG Boost splits up to the maximum depth specified and prunes the tree backward and removes splits beyond which there is an only negative loss.

****
****
## imbalanced-learn API

### Installation

    pip install -U imbalanced-learn

or use Anaconda

    conda install -c conda-forge imbalanced-learn

### [imbalanced-learn API](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.under_sampling)

#### [imblearn.under_sampling](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.under_sampling)

The imblearn.under_sampling provides methods to under-sample a dataset.

* Prototype generation

The _imblearn.under_sampling.prototype_generation_ submodule contains methods that generate new samples in order to balance the dataset.

* Prototype generation

Method | Description
-------|------------
under_sampling.CondensedNearestNeighbour([…])|Class to perform under-sampling based on the condensed nearest neighbour method.
under_sampling.EditedNearestNeighbours([…])|Class to perform under-sampling based on the edited nearest neighbour method.
under_sampling.RepeatedEditedNearestNeighbours([…])|Class to perform under-sampling based on the repeated edited nearest neighbour method.
under_sampling.AllKNN([sampling_strategy, …])|Class to perform under-sampling based on the AllKNN method.
under_sampling.InstanceHardnessThreshold([…])| Class to perform under-sampling based on the instance hardness threshold.
under_sampling.NearMiss([sampling_strategy, …])| Class to perform under-sampling based on NearMiss methods.
under_sampling.NeighbourhoodCleaningRule([…])| Class performing under-sampling based on the neighbourhood cleaning rule.
under_sampling.OneSidedSelection([…])| Class to perform under-sampling based on one-sided selection method.
under_sampling.RandomUnderSampler([…])| Class to perform random under-sampling.
under_sampling.TomekLinks([…])| Class to perform under-sampling by removing Tomek’s links.


#### [imblearn.over_sampling](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling)

Method | Description
-------|------------
over_sampling.ADASYN([sampling_strategy, …])|Perform over-sampling using Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets.
over_sampling.BorderlineSMOTE([…])|Over-sampling using Borderline SMOTE.
over_sampling.KMeansSMOTE([…])|Apply a KMeans clustering before to over-sample using SMOTE.
over_sampling.RandomOverSampler([…])|Class to perform random over-sampling.
over_sampling.SMOTE([sampling_strategy, …])|Class to perform over-sampling using SMOTE.
over_sampling.SMOTENC(categorical_features)|Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC).
over_sampling.SVMSMOTE([sampling_strategy, …])|Over-sampling using SVM-SMOTE.



#### [imblearn.combine](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.combine)

The imblearn.combine provides methods which combine over-sampling and under-sampling.

Method | Description
-------|------------
combine.SMOTEENN([sampling_strategy, …])|Class to perform over-sampling using SMOTE and cleaning using ENN.
combine.SMOTETomek([sampling_strategy, …])|Class to perform over-sampling using SMOTE and cleaning using Tomek links.



#### [imblearn.ensemble](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.ensemble)

The imblearn.ensemble module include methods generating under-sampled subsets combined inside an ensemble.

Method | Description
-------|------------
ensemble.BalanceCascade(**kwargs)|Create an ensemble of balanced sets by iteratively under-sampling the imbalanced dataset using an estimator.
ensemble.BalancedBaggingClassifier([…])|A Bagging classifier with additional balancing.
ensemble.BalancedRandomForestClassifier([…])|A balanced random forest classifier.
ensemble.EasyEnsemble(**kwargs)|Create an ensemble sets by iteratively applying random under-sampling.
ensemble.EasyEnsembleClassifier([…])|Bag of balanced boosted learners also known as EasyEnsemble.
ensemble.RUSBoostClassifier([…])|Random under-sampling integrating in the learning of an AdaBoost classifier.


#### [imblearn.keras: Batch generator for Keras](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.keras)

The imblearn.keras provides utilities to deal with imbalanced dataset in keras.

Method | Description
-------|------------
keras.BalancedBatchGenerator(X, y[, …])|Create balanced batches when training a keras model.
keras.balanced_batch_generator(X, y[, …])|Create a balanced batch generator to train keras model.

#### [imblearn.tensorflow: Batch generator for TensorFlow](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.tensorflow)

The imblearn.tensorflow provides utilities to deal with imbalanced dataset in tensorflow.

Method | Description
-------|------------
tensorflow.balanced_batch_generator(X, y[, …])|Create a balanced batch generator to train keras model.

#### [imblearn.pipeline: Pipeline](https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.pipeline)

The imblearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms, samples and estimators.

Method | Description
-------|------------
pipeline.Pipeline(steps[, memory, verbose])|Pipeline of transforms and resamples with a final estimator
pipeline.make_pipeline(\*steps, \*\*kwargs)|Construct a Pipeline from the given estimators