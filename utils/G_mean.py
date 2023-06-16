from sklearn.metrics import multilabel_confusion_matrix, recall_score,confusion_matrix
from math import sqrt

def G_mean(y_true,y_pred):
    y_true,y_pred = y_true.cpu(),y_pred.cpu()

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    # print(mcm)
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    # recall_positive = tp.sum() / (tp.sum() + fn.sum())
    # print('calc_r:',recall_positive)

    tpr = recall_score(y_true, y_pred, average='micro')
    # print('TPR:', tpr)
    tnr = tn.sum() / (tn.sum() + fp.sum())
    # print('TNR:', tnr)

    g_mean = (tpr * tnr) ** 0.5
    # print('g_mean:', g_mean)
    return g_mean


def G_mean_2(y_true,y_pred):
    cm = confusion_matrix(y_true,y_pred)
    g_mean = sqrt(cm[0][0] / (cm[0][0] + cm[0][1]) * cm[1][1] / (cm[1][0] + cm[1][1]))
    return g_mean


def G_mean_DPA(mcm):
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    tpr = tp.sum() / (tp.sum() + fn.sum())
    # print('calc_r:',recall_positive)
    # print('TPR:', tpr)
    tnr = tn.sum() / (tn.sum() + fp.sum())
    # print('TNR:', tnr)

    g_mean = (tpr * tnr) ** 0.5
    # print('g_mean:', g_mean)
    return g_mean
