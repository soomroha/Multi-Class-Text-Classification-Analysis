import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from clean_data_helper import load_and_clean_data


df = load_and_clean_data()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.TITLE).toarray()
labels = df.category_id


print("The Mean Accuracy and Standard Deviation of the Algorithms")

ml_models = []
ml_models.append(('Logistic Regression', LogisticRegression(random_state=0)))
ml_models.append(('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)))
ml_models.append(('Naive Bayes', MultinomialNB()))
ml_models.append(('Linear SVC', LinearSVC()))

accuracy_results = []
names = []

scoring = 'accuracy'
for name, model in ml_models:

    kfold = model_selection.KFold(n_splits=10, random_state=8)
    cross_val_score_outcome = model_selection.cross_val_score(model, features, labels, cv=kfold, scoring=scoring)
    accuracy_results.append(cross_val_score_outcome)
    names.append(name)
    print("%s Mean Accuracy: %f Standard Deviation: %f" % (name, cross_val_score_outcome.mean(), cross_val_score_outcome.std()))


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(accuracy_results)
ax.set_xticklabels(names)
plt.show()