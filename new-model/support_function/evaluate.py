import numpy as np


def new_accuracy(y_train, second_y_train, y_pred):
    y_train = np.array(y_train)
    second_y_train = np.array(second_y_train)
    y_pred = np.array(y_pred)
    main_label_count = np.sum(y_pred == y_train)
    condition = np.logical_or(y_pred == second_y_train, y_pred == y_train)
    count = np.sum(condition)
    accuracy = count / len(y_train)
    return accuracy, main_label_count, count - main_label_count


## Custom F-score
def precision_recall(first_label, second_label, predicted_labels, average="weighted"):
    pair_label = set()
    for i in range(len(first_label)):
        pair = (
            min(first_label[i], second_label[i]),
            max(first_label[i], second_label[i]),
        )
        pair_label.add(pair)
    # Initialize counters for each class
    TP = {cls: 0 for cls in pair_label}
    TN = {cls: 0 for cls in pair_label}
    FP = {cls: 0 for cls in pair_label}
    FN = {cls: 0 for cls in pair_label}

    for true1, true2, pred in zip(first_label, second_label, predicted_labels):
        tmp_pair = (min(true1, true2), max(true1, true2))
        for cls in pair_label:
            if tmp_pair == cls and pred in cls:  # True Positive (TP)
                TP[cls] += 1
            elif tmp_pair == cls and pred not in cls:  # False Negative (FN)
                FN[cls] += 1
            elif tmp_pair != cls and pred in cls:  # False Positive (FP)
                FP[cls] += 1
            elif tmp_pair != cls and pred not in cls:  # True Negative (TN)
                TN[cls] += 1

    if average == "weighted":
        precision_sum = 0
        recall_sum = 0
        weight_sum = 0
        for cls in pair_label:
            weight = TP[cls] + FN[cls]
            weight_sum += weight
            if TP[cls] + FP[cls] > 0:
                precision_sum += (TP[cls] / (TP[cls] + FP[cls])) * weight
            if TP[cls] + FN[cls] > 0:
                recall_sum += (TP[cls] / (TP[cls] + FN[cls])) * weight
        precision = precision_sum / weight_sum if weight_sum > 0 else 0
        recall = recall_sum / weight_sum if weight_sum > 0 else 0
    elif average == "micro":
        TP_total = sum(TP.values())
        FP_total = sum(FP.values())
        FN_total = sum(FN.values())
        precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
        recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    elif average == "macro":
        precision_sum = 0
        recall_sum = 0
        count = 0
        for cls in pair_label:
            if TP[cls] + FP[cls] > 0:
                precision_sum += TP[cls] / (TP[cls] + FP[cls])
                count += 1
            if TP[cls] + FN[cls] > 0:
                recall_sum += TP[cls] / (TP[cls] + FN[cls])
        precision = precision_sum / count if count > 0 else 0
        recall = recall_sum / count if count > 0 else 0
    return precision, recall


def new_f1_score(first_label, second_label, predicted_labels, average="weighted"):
    precision, recall = precision_recall(
        first_label, second_label, predicted_labels, average
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score
