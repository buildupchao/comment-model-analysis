from data_etl import etl
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from result_display import display

X_train, X_test, y_train, y_test, vectorizer = etl.do_data_etl()

random_forest_classifier = RandomForestClassifier(criterion='entropy',random_state=1,n_jobs=2)
pipeline = make_pipeline(vectorizer, random_forest_classifier)
history = pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

display.display_report(y_test, y_predict)