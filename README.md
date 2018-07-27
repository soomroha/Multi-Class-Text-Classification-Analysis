# Multi-Class-Text-Classification-Analysis

##Loading and cleaning the dataset

```python
df = pd.read_csv('headlines.csv')
df = df[['CATEGORY','TITLE']]
df = df[pd.notnull(df['TITLE'])]
df.columns = ['CATEGORY', 'TITLE']
df.TITLE = df.TITLE.apply(lambda x: x.lower())
df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', '1234567890')))
df['category_id'] = df['CATEGORY'].factorize()[0]
```
## Top 25 Features of the Categories

## Mean Accuracy and Standard Deviation of the Algorithms

Logistic Regression: Mean Accuracy: 0.847214 Standard Deviation: 0.046154
Random Forest: Mean Accuracy: 0.361110 Standard Deviation: 0.098128
Naive Bayes: Mean Accuracy: 0.855489 Standard Deviation: 0.038743
Linear SVC: Mean Accuracy: 0.849241 Standard Deviation: 0.045773


![figure_1](https://user-images.githubusercontent.com/38365732/43301490-7efdc74e-9133-11e8-809f-15925c5b5889.png)

## Confusion Matrices
