import sys
sys.path.append('../')

from data.get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt


X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=3)

knn_classifier.fit(X_train, y_train)
predictions = knn_classifier.predict(X_test)

# Evaluate the classifier
accuracy = knn_classifier.score(X_test, y_test)

# Calculate metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
confusion_matrix_result = confusion_matrix(y_test, predictions)

# Save metrics to file
with open("result/metrics_knn.txt", "w") as f:
    f.write(f'+ precision = {precision:.3f}\n')
    f.write(f'+ recall = {recall:.3f}\n')
    f.write(f'+ f1_score = {f1_score:.3f}\n')
    f.write(f'+ accuracy = {accuracy:.3f}\n')

# Save confusion matrix plot to file
plt.figure(figsize=(8,6))
ax = sns.heatmap(data=confusion_matrix_result, fmt="d", annot=True)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
plt.savefig("result/confusion_matrix_knn.png")  # Save the plot to a file
plt.close()


