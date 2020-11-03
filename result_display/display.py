import matplotlib.pyplot as plt
import prettytable as pt
from sklearn import metrics

"""
|-------|-------|-------|
|       |  YES  |   NO  |
|-------|-------|-------|
|  YES  |  TP   |   FN  |
|-------|-------|-------|
|  NO   |  FP   |   TN  |
|-------|-------|-------|

Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = 2*Recall*Precision/(Recall+Precision)
Accuracy = (TP+TN)/(TP+FN+FP+TN)

"""

def calculate_kpi(y_train, y_predict):
    TP = FP = TN = FN = 0

    for i in range(len(y_predict)):
        if y_predict[i] == 1:
            if y_train[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_train == 1:
                FN += 1
            else:
                TN += 1

    table = pt.PrettyTable(["TN", "FP", "FN", "TP", "precision", "accuracy", "recall", "F1", "roc"])

    accuracy = round(metrics.accuracy_score(y_train, y_predict), 2)
    recall = round(metrics.recall_score(y_train, y_predict), 2)
    f1 = round(metrics.f1_score(y_train, y_predict), 2)
    roc = round(metrics.roc_auc_score(y_train, y_predict), 2)
    pre = round(metrics.precision_score(y_train, y_predict), 2)

    table.add_row([TN, FP, FN, TP, pre, accuracy, recall, f1, roc])
    table.border = False
    print(table)

    return TN, FP, FN, TP

def display_report(y_train, y_predict):
    y_predict = [round(x) for x in y_predict]
    print(metrics.classification_report(y_train, y_predict))

    calculate_kpi(y_train, y_predict)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_train, y_predict)
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([-1, 1], [-1, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

