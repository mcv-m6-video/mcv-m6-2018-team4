import os
import cv2
import numpy as np

def readImages(Path,extension):
    if extension == "" or ( extension != "jpg" and extension != "png" ) or not os.path.exists(Path):
        print "Extension or path is invalid"
        return -1
    if len( os.listdir(Path)) == 0:
        print "Directory is empty"
        return 0

    imgs = []
    for file in sorted(os.listdir(Path)):
        if file.endswith("." + extension):
            print "loading image " +  os.path.join(Path, file)
            image = cv2.imread(os.path.join(Path, file))
            imgs.append(image)
    return imgs



def ConfusionMatrix( GT, Prediction):
    if len(GT) != len(Prediction):
        print "Datasets does not have same size"
        return -1
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    print "Computing Confusion Matrix"
    for img in range(len(GT)):
        GT_img = np.array(GT[img])
        Prediction_img = np.array(Prediction[img])
        print "Computing image: " + str(img)
        for i in range(GT_img.shape[0]):
            for j in range(GT_img.shape[1]):
                GT_value = GT_img[i][j][0]
                Prediction_value = Prediction_img[i][j][0]
                if GT_value >= 170:
                    GT_value = 1
                else:
                    GT_value = 0
                if   1 == Prediction_value and GT_value == 1:
                    TP +=1
                elif 0 == Prediction_value and GT_value == 0:
                    TN +=1
                elif 1 == Prediction_value and GT_value == 0:
                    FP +=1
                elif 0 == Prediction_value and GT_value == 1:
                    FN +=1
    return [TP,TN,FP,FN]

def Metrics(TP,TN,FP,FN):
    print "Computing Metrics with TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN)
    Accuracy   = ( float(TP) + float(TN) ) / ( float(TP) + float(TN) + float(FP) + float(FN) ) 
    Recall     = float(TP) / ( float(TP) + float(FN) ) 
    Precision  = float(TP) / ( float(TP) + float(FP) )
    F1         = 2 * float(Precision) * float(Recall) / ( float(Precision) + float(Recall) )
    print "Accuracy: "   + str(Accuracy   * 100)
    print "Recall: "     + str(Recall     * 100)
    print "Precision: "  + str(Precision  * 100)
    print "F1: "         + str(F1         * 100)
    return Accuracy,Recall,Precision,F1



def ROCAndPRCurves(descriptors,label_per_descriptor,classifier):
    print "Plotting ROC curve"
    tprs = []
    aucs = []
    LROC  = []
    for i in range(len(descriptors)):
        LROC.append(int(GetKey(label_per_descriptor[i])))
    LROC = np.array(LROC)
    #Binarize to convert in one vs all
    LROC = label_binarize(LROC, classes=[1,2,3,4,5,6,7,8])
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test =\
    train_test_split(descriptors, LROC, test_size=0.125, random_state=0)
    # classifier
    clf = OneVsRestClassifier(classifier)
    probabilities = clf.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(8)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(8):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 8

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','r', 'g', 'b', 'y','m'])
    for i, color in zip(range(8), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    print "Plotting Precision/Recall curve"
    # Compute PR curve for each class
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(8):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            probabilities[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], probabilities[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        probabilities.ravel())
    average_precision["micro"] = average_precision_score(y_test, probabilities,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(8), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()