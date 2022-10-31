---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Homework 4: Preprocessing 

## Introduction

A crucial step when using machine learning algorithms on real-world datasets is preprocessing. This homework will give you some practice of data preprocessing and building a supervised machine learning pipeline on a real-world dataset. 


## Imports 

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
```

## Exercise A: Introducing the dataset
<hr>

In this lab, you will be working on [the adult census dataset](https://www.kaggle.com/uciml/adult-census-income#). 
Download the CSV and save it as `adult.csv` under this homework folder. 

This is a classification dataset and the classification task is to predict whether income exceeds 50K per year or not based on the census data. You can find more information on the dataset and features [here](http://archive.ics.uci.edu/ml/datasets/Adult).

The starter code below loads the data CSV (assuming that it is saved as `adult.csv` in this folder). 

_Note that many popular datasets have sex as a feature where the possible values are male and female. This representation reflects how the data were collected and is not meant to imply that, for example, gender is binary._

```python slideshow={"slide_type": "slide"}
census_df = pd.read_csv("adult.csv")
census_df.shape
```

### Data splitting 
rubric={points:5}

In order to avoid violation of the golden rule, the first step before we do anything is splitting the data. 

**Your tasks:**

1. Split the data into `train_df` (80%) and `test_df` (20%) with `random_state = 24`. Keep the target column (`income`) in the splits so that we can use it in the exploratory data analysis.  

_Usually having more data for training is a good idea. But here I'm using 80%/20% split because this is kind of a big dataset for a modest laptop. A smaller training data means it won't take too long to train the model on your laptop. A side advantage of this would be that with a bigger test split, we'll have a more reliable estimate of the deployment performance!_

```python deletable=false nbgrader={"cell_type": "code", "checksum": "664609bb3239b1f2db201db3a084249e", "grade": true, "grade_id": "cell-ede84e17a177c40c", "locked": false, "points": 5, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "slide"}
train_df = None
test_df = None

# BEGIN YOUR CODE HERE
train_df, test_df = train_test_split(census_df, test_size=0.20, random_state=24)

# END YOUR CODE HERE
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise B: Exploratory data analysis (EDA)
<hr>
<!-- #endregion -->

Let's examine our `train_df`. 

```python
train_df.sort_index()
```

We see some missing values represented with a "?". Probably these were the questions not answered by some people during the census.  Usually `.describe()` or `.info()` methods would give you information on missing values. But here, they won't pick "?" as missing values as they are encoded as strings instead of an actual NaN in Python. So let's replace them with `np.nan` before we carry out EDA. If you do not do it, you'll encounter an error later on when you try to pass this data to a classifier. 

```python
train_df_nan = train_df.replace("?", np.nan)
test_df_nan = test_df.replace("?", np.nan)
train_df_nan.shape
```

```python
# train_df_nan.sort_index()
train_df_nan
```

The "?" symbols are now replaced with `NaN` values. 


### Visualizing features
rubric={points:10}


#### Task B1
rubric={points:4}

`display` the information given by `train_df_nan.info()` and `train_df_nan.describe()` methods. 
In the case of `.describe()`, you can **optimally** use the `include="all"` argument to show summary statistics of all  features. 

```python deletable=false nbgrader={"cell_type": "code", "checksum": "b2743989e79d79206999089ddfe69e55", "grade": true, "grade_id": "cell-30f3e2214cfdbf33", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE

from IPython.display import display

print(".info() output \n")

display(train_df_nan.info())

print("\n")

print(".describe() output \n")
display(train_df_nan.describe(include="all"))

# END YOUR CODE HERE
```

#### Task B2 
rubric={points:6}

Visualize the histograms of numeric features 

Hint: use `dataframe.hist` to show the distribution of six numeric features.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "e26c83a5f396d331afce9aa64682664b", "grade": true, "grade_id": "cell-440f98becf9bfe9a", "locked": false, "points": 6, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
from matplotlib.pyplot import figure
# ax = train_df_nan.hist(column=["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"],figsize=(20,20),sharey=True)
# ax = ax[0]
# # for x in ax:
# #     print(x)
# #     x.set_title("")
# #     x.set_xlabel("Session Duration (Seconds)", labelpad=20, weight='bold', size=12)
# #     x.set_ylabel("Sessions", labelpad=20, weight='bold', size=12)

fig,axes = plt.subplots(3,2)
fig.set_size_inches(15,15)
cols = ["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]
axes = axes.ravel()
for i,j in enumerate(axes):
    j.hist(train_df_nan[cols[i]],edgecolor="white")
    j.set_xlabel(cols[i],labelpad=5, weight='bold', size=12)
    j.set_ylabel("Number of People",labelpad=5, weight='bold', size=12)

# END YOUR CODE HERE
```

#### Task B3
rubric={points:4}

From the visualizations, which features seem relevant for the given prediction task?(You can pick multi-features).

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "e7b871c4fc21893b77345f6b8679f02a", "grade": true, "grade_id": "cell-42db519234eb9146", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>
For the given task, education and number of hours worked per week are more relevant than other features such as capital gain/loss. Assuming that the education.num refers to number of years spent on education, we can safely say that higher number will mean higher wage. Similarly, the higher number of work hours depening upon the hourly wage, may also indicate if someone earns more than 50,000 dollars per year.
Other feature, such as capital gain/loss are hard to use for this task without any additional information. Features such as  age are of no use, since the younger person with no eudcation may not earn as much as an old person with high school diploma. The feaure age may be used with other feature such as how may years in school/education, how many hours do they work and so on.
<!-- #endregion -->

### Identify transformations to apply

rubric={points:20}



#### Task B4
rubric={points:13}

Identify what kind of feature transformations (`scaling`, `imputation`, `one hot encoding`) you would apply on each column in the dataset and fill in the table below accordingly. You may decide to apply any transformations on a certain column or entirely drop a column from your model. That's totally fine. 

As an example, we use imputation and One-Hot encoding for feature `occupation` here.


| Feature | Transformation |
| --- | ----------- |
| occupation | imputation, One-Hot Encoding |
| age | Scaling |
| workclass | One-Hot Encoding |
| fnlwgt | Scaling |
| education | One-Hot Encoding |
| education.num | Imputation,Scaling |
| marital.status | One-Hot Encoding |
| relationship | One-Hot Encoding |
| race | Drop |
| sex | One-Hot Encoding |
| capital.gain | Scaling |
| capital.loss | Scaling |
| hours.per.week | Scaling |
| native.country | One-Hot Encoding |


#### Task B5
rubric={points:5}

Identify different feature types for applying different transformations. 
In particular, fill in the lists below.

Hint:
1. This dataset is very special - the features with missing values are categorical. So we don't create a list for `imputation_features`.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "f7110445da3c456374c9de11648e68de", "grade": true, "grade_id": "cell-a19ac474c1373859", "locked": false, "points": 5, "schema_version": 3, "solution": true, "task": false}
# It's OK to keep some of the lists empty or add new lists.
numeric_features = [] 
categorical_features = [] 
ordinal_features = [] 
binary_features = [] 
drop_features = []  # do not include these features in modeling
passthrough_features = [] # do not apply any transformation

# Example: numeric_features = ["age"] 
# BEGIN YOUR CODE HERE

numeric_features = ["age","fnlwgt","capital.gain","capital.loss","education.num","hours.per.week"]  
categorical_features = ["occupation","education","relationship","marital.status"] 
ordinal_features = ["marital.status"] 
binary_features = [] 
drop_features = ["native.country","race","sex"]  # do not include these features in modeling
passthrough_features = ["native.country","race","sex"] # do not apply any transformation
# END YOUR CODE HERE


target = "income"

```

#### Task B6
rubric={points:2}

Is including the `race` feature for predicting income ethically a good idea? Briefly discuss.

Hint:
1. This question is a bit open-ended and there is no single correct solution.

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "56c5838ee59fb77a273b721c8d1a0eaf", "grade": true, "grade_id": "cell-e1334ca6da3801d2", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>
In reality, most jobs/occupation are related to the socio-economic status of an individual. For example, a large section of particualr community may be involved in a certain occupation. Although, based on this simple fact, we cannot/should not assume the occupation of each and every member of the said community. Therefore, I believe, it is not a good idea to include "race" feature for predicting the income.
<!-- #endregion -->

### Separating feature vectors and targets  
rubric={points:6}

<br>


#### Task B7
rubric={points:4}

Create `X_train`, `y_train`, `X_test`, `y_test` from `train_df_nan` and `test_df_nan`.

Hint:
1. `income` is the target.
2. The rest are considered as features.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "3ed9e4e497355b656151201c061c9f7c", "grade": true, "grade_id": "cell-e0948492f2ad7018", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
# Split the dataset, feature/target:
# BEGIN YOUR CODE HERE

X_train = train_df_nan.drop(columns = ["income"])
y_train = train_df_nan["income"]

X_test = test_df_nan.drop(columns = ["income"])
y_test = test_df_nan["income"]

# END YOUR CODE HERE
```

#### Task B8
rubric={points:2}

At this point, if you train kNN model on `X_train` and `y_train`, would it work? Why or why not?

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "bd6387ed6f904f23e5c9337bd26f2f49", "grade": true, "grade_id": "cell-f0a5e4962c33bbdf", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>
kNN model does not work well when the dataset has too many dimensions/features. Moreover, our data contains too many categorical features and numerical features have missgin data. If I were to train kNN model, right now, it would give me appropriate results.
<!-- #endregion -->

## Exercise C: Preprocessing
<hr>


### Preprocessing using `sklearn`'s `ColumnTransformer` and `Pipeline`
rubric={points:18}

Let's carry out preprocessing using `sklearn`'s `ColumnTransformer` and `Pipeline`. Note that you can define pipelines in two ways: 
- by using [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and explicitly providing named steps
- by using [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline), which automatically names the steps in the pipeline with their class names. 

Similarly you can create a column transformer in two ways:
- by using [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- by using [`make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html) 

You may use the method of your choice but `make_pipeline` and `make_column_transformer` are highly recommended.  


#### Task C1 
rubric={points:10}

Create a column transformer `preprocessor` based on transformations you want to apply on the data from [Task B5](#Task-B5).

Hint
1. There are several features with missing values. Fortunately, they are categorical features.
2. Don't forget add `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder` for your `categorical_features`.
3. You can use `make_pipeline` to combine `SimpleImputer` and `OneHotEncoder`.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "aa3ab60f86fd234e74babf884e24175d", "grade": true, "grade_id": "cell-d8d20476ac487747", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false}
preprocessor = None
# BEGIN YOUR CODE HERE

numeric_features = ["age","fnlwgt","capital.gain","capital.loss","education.num","hours.per.week"]  
categorical_features = ["occupation","education","relationship","marital.status"] 
ordinal_features = ["marital.status"] 
binary_features = [] 
drop_features = ["native.country","race","sex"]  # do not include these features in modeling
passthrough_features = ["native.country","race","sex"]

numeric_transformer = make_pipeline(StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())

preprocessor = make_column_transformer((numeric_transformer,numeric_features),(categorical_transformer, categorical_features),("drop",drop_features))
# preprocessor
# END YOUR CODE HERE

```
#### Task C2 
rubric={points:4}

Transform the data by calling `fit_transform` on the training set. 
Then **print** or **display** the shape of the transformed data.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "8105aa5c74d1b3a047ff050b306ff2f4", "grade": true, "grade_id": "cell-3ef662c3ea19f095", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
transf = preprocessor.fit_transform(X_train)
print(f"The shape of the transformed data is {transf.shape}")
# END YOUR CODE HERE
```

<!-- #region -->
#### Task C3
rubric={points:4}


Why do we need to use a column transformer in this case? Briefly explain.
<!-- #endregion -->

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "858862684b37ecfad39c4e5de30f1e18", "grade": true, "grade_id": "cell-7309c309f7fa5f78", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>
In our data, we have different type of features such as categorical, numerical, and ordinal. In order to develop a supervised machine learning pipeline here we need to apply different transformation on different columns/features. However, it is tedious to apply transformation to each of the features one by one.
Thus, we use column transformer to build all our transformation into a single object. This way we can ensure that we do the same operations to all splits of the data.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise D: Building models
<hr>

Now that we have preprocessed features, we are ready to build models. Below, I'm providing the function we used in class which returns mean cross-validation score along with standard deviation for a given model. Feel free to use it to keep track of your results if you like. 
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
results_dict = {} # dictionary to store all the results
```

```python

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Baseline model 
rubric={points:6}
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
#### Task D1
rubric={points:3}

Define a **pipeline** with two steps: `preprocessor` from [Task C2](Task-C2) and `scikit-learn`'s `DummyClassifier` with `strategy="prior"` as your classifier.
<!-- #endregion -->

```python deletable=false nbgrader={"cell_type": "code", "checksum": "7f17a77b2ed561e4c651d47ebe5da2bd", "grade": true, "grade_id": "cell-f21d59b77ebc9be9", "locked": false, "points": 3, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", DummyClassifier(strategy="prior"))])

# END YOUR CODE HERE
```

<!-- #region slideshow={"slide_type": "slide"} -->
#### Task D2
rubric={points:3}

Carry out 5-fold cross-validation with the pipeline. Store the results in `results_dict['dummy]` above. 

> You may use the function `mean_std_cross_val_scores` above to carry out cross-validation and storing results. Refer to the class notes if you are unsure about how to use it. 
<!-- #endregion -->

```python deletable=false nbgrader={"cell_type": "code", "checksum": "fa2bef41178a4d2b1e8c59652e845a3b", "grade": true, "grade_id": "cell-00cee76d80b6e733", "locked": false, "points": 3, "schema_version": 3, "solution": true, "task": false}
results_dict["dummy"] = None
# BEGIN YOUR CODE HERE
results_dict["dummy"] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)

# END YOUR CODE HERE
```

Now let us show the results.

```python
pd.DataFrame(results_dict).T
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Trying different classifiers
rubric={points:12}
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
#### Task D3
rubric={points:5}

For each of the models (`DecisionTreeClassifier` and `KNeighborsClassifier`) in the starter code below:

- Define a pipeline with two steps: `preprocessor` from [Task C2](#Task-C2) and the model as your classifier. 
- Carry out 5-fold cross-validation with the pipeline.  
- Store the results in `results_dict`. 
    
<!-- #endregion -->

```python
models = {
    "decision tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
}
```
```python deletable=false nbgrader={"cell_type": "code", "checksum": "6dc70200605d7cba32ceb40ca3f964b6", "grade": true, "grade_id": "cell-4a60eadeeea02f19", "locked": false, "points": 5, "schema_version": 3, "solution": true, "task": false}
np.random.seed(12345)
# BEGIN YOUR CODE HERE
# pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", models[i])])
pipe = [0]*2
for i,j in enumerate(models):
#     print(i)
#     variable_name = "pipe_" + str(i)
#     if variable_name == "pipe_decision tree":
#         variable_name = "pipe_decision"
#     print(variable_name)
    pipe[i] = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", models[j])])
    results_dict[j] = mean_std_cross_val_scores(pipe[i], X_train, y_train, cv=5, return_train_score=True)

# pipe_DTC = pipe[0]
# pipe_kNN = pipe[1]

# END YOUR CODE HERE
```

<!-- #region slideshow={"slide_type": "slide"} -->
#### Task D4
rubric={points:2}

Display all the results so far as a pandas dataframe.
<!-- #endregion -->

```python deletable=false nbgrader={"cell_type": "code", "checksum": "bee5c29613d9f592b8f818d803e2cabe", "grade": true, "grade_id": "cell-ecc02b55d417c46e", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
pd.DataFrame(results_dict).T

# END YOUR CODE HERE
```

### Exploring importance of scaling
rubric={points:10}

In this exercise you'll examine whether scaling helps in case of KNNs. 


#### Task D6
rubric={points:4}

Create a column transformer **without** the `StandardScaler` step for `numeric_features`. 
You can refer your to [Task C1](#Task-C1).

```python deletable=false nbgrader={"cell_type": "code", "checksum": "264f82722e13e823794282e4d89b24ab", "grade": true, "grade_id": "cell-bb06df720e04ebd4", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
numeric_transf = make_pipeline(SimpleImputer())
categorical_transf = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())

preprocessor_d = make_column_transformer((numeric_transf,numeric_features),(categorical_transf, categorical_features),("drop",drop_features))


# END YOUR CODE HERE
```

#### Task D7
rubric={points:4}

Repeat the steps in [Task D3](#Task-D3) with this new column transformer. Save all results in `results_dic_compare`.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "bace5328bb1a41629bad752c462cd13d", "grade": true, "grade_id": "cell-51318be5ac76c162", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
np.random.seed(12345)
results_dict_compare = {}  # dictionary to store all the results

# BEGIN YOUR CODE HERE
pipe_d = [0]*2
for i,j in enumerate(models):
#     print(i)
#     variable_name = "pipe_" + str(i)
#     if variable_name == "pipe_decision tree":
#         variable_name = "pipe_decision"
#     print(variable_name)
    pipe_d[i] = Pipeline(steps=[("preprocessor", preprocessor_d), ("classifier", models[j])])
    results_dict_compare[j] = mean_std_cross_val_scores(pipe_d[i], X_train, y_train, cv=5, return_train_score=True)


# END YOUR CODE HERE
```

```python
pd.DataFrame(results_dict_compare).T # compare the result with the outcome without doing feature scaling   
```

#### Task D8
rubric={points:2}

Compare the results of scaled numeric features with unscaled numeric features. 
1. Is scaling necessary for decision trees? (Yes/No)
2. Is scaling necessary for knn? (Yes/No)

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "c7eb1956252a1274e0f77af2bbb82939", "grade": true, "grade_id": "cell-e0a33e852b68eeff", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>

1. No. Unlike kNN, Decision trees use rule based approach rather than distance based appraoch. Therefore, in case of Decision trees, we do not require scaling.

2. Yes. Since kNN uses euclidean distance measure, scaling of numerical feature plays an important role. Hence the answer is Yes. Scaling is necessary for kNN. 
<!-- #endregion -->

### Hyperparameter optimization
rubric={points:8}

In this exercise, you'll carry out hyperparameter optimization for the hyperparameter `n_neighbors` of KNeighborsClassifier. 
In practice you'll carry out hyperparameter optimization for all different hyperparameters for the most promising classifiers. 
For the purpose of this assignment, we'll only do it for the K Neighbors classifier with one hyperparameter: `n_neighbors`. 


#### Task D9
rubric={points:4}

For each `n_neighbors` value in the `param_grid` in the starter code below: 
- Create a pipeline object with two steps: `preprocessor` from [Task C1](#Task-C1) and KNeighbors classifier with the value of `n_neignbors`.
- Carry out 5-fold cross validation with the pipeline using the function `mean_std_cross_val_scores`.  
- Store the results in `results_dict_hyper` where the key is the `n_neignbors` value and the value is times and scores.
- This step take a few minutes as you are training 4 knn models with 5 fold cross-validation.
- Display results as a pandas DataFrame.
    

```python
param_grid = [2, 3, 4, 5]
param_grid
```

```python deletable=false nbgrader={"cell_type": "code", "checksum": "d15cfcef1c387f93deab5191d04fffe3", "grade": true, "grade_id": "cell-f88c2042ea20cd24", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}
results_dict_hyper = {}

# BEGIN YOUR CODE HERE
for i,j in enumerate(param_grid):
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=j))])
    results_dict_hyper[i] = mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)
# END YOUR CODE HERE

pd.DataFrame(results_dict_hyper).T
```

You can find the best hyper-parameter from `[2, 3, 4, 5]`. Keep it in your mind.

<!-- #region slideshow={"slide_type": "slide"} tags=[] -->
## Exercise E: Evaluating on the test set
<hr>

Now that we have a best performing model, it's time to assess our model on the set aside test set. In this exercise you'll examine whether the results you obtained using cross-validation on the train set are consistent with the results on the test set. 
<!-- #endregion -->

### Task E1
rubric={points:3}

Train the best performing model on **the entire training set** with the pipeline define from [Task D9](#Task-D9).

Hint:
1. Cross validation is no longer needed as you have the best hyper-parameters
2. You build a knn model with `X_train` and `y_train`.

```python deletable=false nbgrader={"cell_type": "code", "checksum": "1845c02c0dfe7b292f80758a05d868a5", "grade": true, "grade_id": "cell-7f2661e2f5b8f4a6", "locked": false, "points": 3, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=4))])
pipe.fit(X_train,y_train)
# END YOUR CODE HERE
```

### Task E2
rubric={points:2}

Report the prediction of this knn model on `X_test`. 

```python deletable=false nbgrader={"cell_type": "code", "checksum": "8b0639f7561e4081379c1764176371c0", "grade": true, "grade_id": "cell-1dc585847b88c884", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}
# BEGIN YOUR CODE HERE
X_test_predictions = pipe.predict(X_test)
display(X_test_predictions) 
# END YOUR CODE HERE
```

We can evaluate the trained model with `accuracy_score` which means the accurate prediction rate.

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, X_test_predictions)
print(accuracy)
```

### Task E3
rubric={points:2}

Are the cross-validation results and test results consistent? (Yes or No)

<!-- #region deletable=false nbgrader={"cell_type": "markdown", "checksum": "4a79a6c0d515a40917c86448f7671529", "grade": true, "grade_id": "cell-3a932e8826c11445", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false} -->
<font color='red'>ANSWER</font>
Yes. The answer from Task D9 for n_neighbors = 4 and the answer from the Task E3 for same number of neighbors matches. Hence, I can say that the cross-validation results and the test resutlts are consistent.
<!-- #endregion -->

## Submission instructions 

**PLEASE READ:** When you are ready to submit your assignment do the following:

1. Run all cells in your notebook to make sure there are no errors by doing `Kernel -> Restart Kernel and Clear All Outputs` and then `Run -> Run All Cells`. 
2. Notebooks with cell execution numbers out of order or not starting from “1” will have marks deducted. Notebooks without the output displayed may not be graded at all (because we need to see the output in order to grade your work).
3. Upload the assignment at Canvas. 
4. Finish the corresponding reflection survey.
