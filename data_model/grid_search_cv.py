from data_etl import etl
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from result_display import display

X_train, X_test, y_train, y_test, vectorizer = etl.do_data_etl()

tfidf = TfidfTransformer()
svc = SVC()
pipe_svc = Pipeline([("scl", vectorizer), ('tfidf',tfidf), ("clf", svc)])
para_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
para_grid = [
    {
        'clf__C':para_range,
        'clf__kernel':['linear']
    },
    {
        'clf__gamma':para_range,
        'clf__kernel':['rbf']
    }
]
gs = GridSearchCV(estimator=pipe_svc, param_grid=para_grid, cv=10, n_jobs=-1)
gs.fit(X_train, y_train)
gs.best_estimator_.fit(X_train, y_train)
y_predict = gs.best_estimator_.predict(X_test)

display.display_report(y_train, y_predict)