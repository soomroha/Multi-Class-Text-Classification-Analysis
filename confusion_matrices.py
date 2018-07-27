import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from clean_data_helper import load_and_clean_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def plot_matrix(matrix, classifier_type):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title('Confusion matrix of ' + classifier_type + '\n')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ["Health", "Entert.", "Science", "Business"])
    ax.set_yticklabels([''] + ["Health", "Entert.", "Science", "Business"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



print("Displaying Confusion Matrices one by one...")

df = load_and_clean_data()
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf_vectorizer.fit_transform(df.TITLE).toarray()
labels = df.category_id
category_id_df = df[['CATEGORY', 'category_id']].drop_duplicates().sort_values('category_id')

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.8, random_state=0)

model_MNB = MultinomialNB()
model_MNB.fit(X_train, y_train)
y_prediction = model_MNB.predict(X_test)
confusion_matrix_MNB = confusion_matrix(y_test, y_prediction)
plot_matrix(confusion_matrix_MNB, "MultinomialNB")

model_linearSVC = LinearSVC()
model_linearSVC.fit(X_train, y_train)
y_prediction = model_linearSVC.predict(X_test)
confusion_matrix_linearSVC = confusion_matrix(y_test, y_prediction)
plot_matrix(confusion_matrix_linearSVC, "LinearSVC")


model_LR = LogisticRegression(random_state=0)
model_LR.fit(X_train, y_train)
y_prediction = model_LR.predict(X_test)
confusion_matrix_LR = confusion_matrix(y_test, y_prediction)
plot_matrix(confusion_matrix_LR, "LogisticRegression")

model_RFC = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
model_RFC.fit(X_train, y_train)
y_prediction = model_RFC.predict(X_test)
confusion_matrix_RFC = confusion_matrix(y_test, y_prediction)
plot_matrix(confusion_matrix_RFC, "RandomForestClassifier")