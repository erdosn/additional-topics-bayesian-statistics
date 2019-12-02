
# Bayesian Stats
* Why does prior probability matter? 


### Objectives
* Use pandas to manipulate data and solve probabilities
* Apply Bayes formula to solve conditional probability problems

### Outline
* Review Baye's Theorem
* Multi-Arm Bandit
* Example
* Activities

# Bayes Formula

**P(X|Y) = Probability of X, given Y.**

![](images/bayes-formula.png)

### Multi Arm Bandit

![](images/bandit1.png)

# Benefit
![](images/bandit2.jpg)

# Question
A fair dice is rolled. What is the probability that it is a 2, given that it is even?


$$\text{P(2|even)} = \frac{\text{P(even|2)P(2)}}{\text{P(even)}}$$

$$\text{P(2|even)} = \frac{\text{(1.0)(1/6)}}{\text{(1/2)}}$$

$$\text{P(2|even)} = 0.333...$$

$$\text{P(2|even)} = 33\%$$

### Activity

### Scenario 1
A family has three children. Assuming that boys and girls are equally likely, determine the probability that the family has...

A) Two boys and one girl GIVEN the first child is a girl.

B) Two girls GIVEN that at least one is a girl

C) Three girls GIVEN that the youngest one is a girl


```python
# P(2 boys and 1 girl| 1=girl)

# P(1=girl | 2 boys and 1 girl)
p_1_girl_2_boys_1_girl = 1/3

# P(2 boys and 1 girl)
p_2_boys_1_girl = 3/8

# P(1=girl)
p_1_girl = 1/2

p_1_girl_2_boys_1_girl*p_2_boys_1_girl/p_1_girl
```




    0.25




```python
# P(2 girls | 1 is a girl)

# P(1 is a girl | 2 girls)
p_1_girl_given_2_girls = 1.0

# P(2 girls)
p_2_girls = 3/8

# GGB
# GBG
# BGG

# P(1 is a girl)
p_1_girl = 7/8

# BBG
# BGB
# GBB
p_1_girl_given_2_girls*p_2_girls / p_1_girl
```




    0.42857142857142855




```python
# C) Three girls GIVEN that the youngest one is a girl

# P(3 girls | 3 = girl)

# P(3 = girl | 3 girls)
p_3_girl_given_3_girls = 1.0


# P(3 = girl)
p_3_girls = 1/8

# P(3 girls)
p_3_girl = 1/2

p_3_girl_given_3_girls*p_3_girls / p_3_girl
```




    0.25



### Scenario 2

You are given an array of points and their labels (see below).  A point is chosen at random. What is the probability that the point is less than 5, given that it is a 0?


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("data/bayes_data.csv")
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>array</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8.206975</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5.543411</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>6.127242</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
p_less_than_5_given_0 = None

# P(0 | given less than 0.5)
total_zeros = df.loc[df['labels']==0, 'array'].values
zeros_less_than_5 = total_zeros[np.where(total_zeros<5.0)]
p_0_given_less_than_5 = zeros_less_than_5.shape[0] / total_zeros.shape[0]


# P(less than 5)
less_than_5_all = df[df['array']<5]['array'].values
p_less_than_5 = less_than_5_all.shape[0] / df.shape[0]


# P(0)
p_zero = 0.50

solution = p_0_given_less_than_5 * p_less_than_5 / p_zero
solution
```




    0.0368



### Scenario 3

Using the iris dataset from sklearn, what is the probability that a flower has a sepal length greater than 6.3, given the flower is a Iris-Versicolour (label=1)?


```python
print(iris.DESCR)
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...



```python
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

df = pd.DataFrame(data, columns=feature_names)
df['label'] = target

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python
# Using the iris dataset from sklearn, what is the probability 
# that a flower has a sepal length greater than 6.3, 
# given the flower is a Iris-Versicolour (label=1)?
# P(s1>6.3 | 1)


# P(1|sl>6.3)
all_greater_than_63 = df[df['sepal length (cm)']>6.3]['label'].values
ones_greater_than_63 = all_greater_than_63[np.where(all_greater_than_63==1)]
p_1_given_greater_63 = ones_greater_than_63.shape[0]/all_greater_than_63.shape[0]


# P(1)
ones = df[df['label']==1]['label'].values
p_one = ones.shape[0]/df.shape[0]

# P(sl>6.3)
p_greater_than_63 = all_greater_than_63.shape[0]/df.shape[0]

p_1_given_greater_63*p_one / p_greater_than_63
```




    0.3117913832199546


