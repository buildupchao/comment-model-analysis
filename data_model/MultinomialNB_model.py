from sklearn.naive_bayes import MultinomialNB

from data_etl import etl
from sklearn.pipeline import make_pipeline
from result_display import display
'''
使用CountVectorizer进行特征提取，使用MultinomialNB分类训练
'''
X_train, X_test, y_train, y_test, vectorizer = etl.do_data_etl()
classifier = MultinomialNB()
pipeline = make_pipeline(vectorizer, classifier)
history = pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)
display.display_report(y_test, y_predict)
