import sys
sys.path.append('../')

from sklearn.naive_bayes import GaussianNB
from data.get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)
accuracy = rf_classifier.score(X_test, y_test)

# Calculate metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
confusion_matrix_result = confusion_matrix(y_test, predictions)

# Save metrics to file
with open("result/metrics_rf.txt", "w") as f:
    f.write(f'+ precision = {precision:.3f}\n')
    f.write(f'+ recall = {recall:.3f}\n')
    f.write(f'+ f1_score = {f1_score:.3f}\n')
    f.write(f'+ accuracy = {accuracy:.3f}\n')

# Save confusion matrix plot to file
plt.figure(figsize=(8,6))
ax = sns.heatmap(data=confusion_matrix_result, fmt="d", annot=True)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
plt.savefig("result/confusion_matrix_rf.png")  # Save the plot to a file
plt.close()
