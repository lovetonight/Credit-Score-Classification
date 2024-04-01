import sys
sys.path.append('../')

from data.get_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  # Import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # Thay thế KNeighborsClassifier bằng GradientBoostingClassifier
# gb_classifier = GradientBoostingClassifier() 

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5]
# } 

# gb_classifier.fit(X_train, y_train)
# predictions = gb_classifier.predict(X_test)

# Evaluate the classifier
# accuracy = gb_classifier.score(X_test, y_test)

# # Calculate metrics
# precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
# confusion_matrix_result = confusion_matrix(y_test, predictions)

# # Save metrics to file
# with open("result/metrics_gb.txt", "w") as f:
#     f.write(f'+ precision = {precision:.3f}\n')
#     f.write(f'+ recall = {recall:.3f}\n')
#     f.write(f'+ f1_score = {f1_score:.3f}\n')
#     f.write(f'+ accuracy = {accuracy:.3f}\n')

# # Save confusion matrix plot to file
# plt.figure(figsize=(8,6))
# ax = sns.heatmap(data=confusion_matrix_result, fmt="d", annot=True)
# ax.set_xlabel("Predicted Labels")
# ax.set_ylabel("True Labels")
# plt.savefig("result/confusion_matrix_gb.png")  # Save the plot to a file
# plt.close()



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Define the classifier
xgb_classifier = XGBClassifier()

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_xgb_classifier = XGBClassifier(**best_params)
best_xgb_classifier.fit(X_train, y_train)

# Make predictions
predictions = best_xgb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
