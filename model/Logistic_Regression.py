import sys
sys.path.append('../')

from data.get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

logistic_classifier = LogisticRegression(solver='lbfgs', max_iter=2000)
logistic_classifier.fit(X_train, y_train)
predictions = logistic_classifier.predict(X_test)
accuracy = logistic_classifier.score(X_test, y_test)


# Calculate metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
confusion_matrix_result = confusion_matrix(y_test, predictions)

# Save metrics to file
with open("result/metrics_logistic.txt", "w") as f:
    f.write(f'+ precision = {precision:.3f}\n')
    f.write(f'+ recall = {recall:.3f}\n')
    f.write(f'+ f1_score = {f1_score:.3f}\n')
    f.write(f'+ accuracy = {accuracy:.3f}\n')

# Save confusion matrix plot to file
plt.figure(figsize=(8,6))
ax = sns.heatmap(data=confusion_matrix_result, fmt="d", annot=True)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
plt.savefig("result/confusion_matrix_logistic.png")  # Save the plot to a file
plt.close()
