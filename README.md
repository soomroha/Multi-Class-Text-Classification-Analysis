# Multi-Class-Text-Classification-Analysis

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
