# Multi-Class-Text-Classification-Analysis

## Project: Analyze the performance of algorithms that classify news headlines into 4 classes.

## Data: [UC Machine Learning Repository News Headlines](https://archive.ics.uci.edu/ml/datasets/News+Aggregator "UC Machine Learning Repository News Headlines")

   • Input: TITLE    
              o	Example: " Bitcoin exchange seeks U.S. bankruptcy protection"
    
   • Output of classification algorithm: CATEGORY   
              o	Example: Business

## Loading and cleaning the dataset

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
## Top 5 Features by Category


![figure_1](https://user-images.githubusercontent.com/38365732/43302586-72830d48-9139-11e8-87e7-b49f11dcf983.png)

## Mean Accuracy and Standard Deviation of the Algorithms

Logistic Regression: Mean Accuracy: 0.847214 Standard Deviation: 0.046154  

Random Forest: Mean Accuracy: 0.361110 Standard Deviation: 0.098128  

Naive Bayes: Mean Accuracy: 0.855489 Standard Deviation: 0.038743  

Linear SVC: Mean Accuracy: 0.849241 Standard Deviation: 0.045773   

![figure_1](https://user-images.githubusercontent.com/38365732/43301490-7efdc74e-9133-11e8-809f-15925c5b5889.png)

## Confusion Matrices of the Algorithms


![figure_1](https://user-images.githubusercontent.com/38365732/43303754-4270a5c4-913f-11e8-8a36-752858ca63ef.png)

![figure_1](https://user-images.githubusercontent.com/38365732/43303790-6d9979e2-913f-11e8-927e-fa4d7be2a3bc.png)

![figure_1](https://user-images.githubusercontent.com/38365732/43303814-8814826c-913f-11e8-813b-3dfd988de47d.png)

![figure_1](https://user-images.githubusercontent.com/38365732/43303855-afda5682-913f-11e8-8e00-a4d4e1456720.png)

References:

https://towardsdatascience.com/a-production-ready-multi-class-text-classifier-96490408757
https://buhrmann.github.io/tfidf-analysis.html
