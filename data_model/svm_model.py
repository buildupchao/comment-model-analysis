from data_etl import etl
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from result_display import display

X_train, X_test, y_train, y_test, vectorizer = etl.do_data_etl(using_test_dataset=False)

svc_cl = SVC()
pipeline = make_pipeline(vectorizer, svc_cl)
pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

display.display_report(y_train, y_predict)