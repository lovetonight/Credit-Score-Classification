import sys

sys.path.append('../')

from data.get_data import get_data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam  # Modified import
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 10  # Kích thước batch
num_classes = 5  # 10 lớp
epochs = 10    # Số epoches

# Data
X, y = get_data()
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model
model = Sequential()
# input shape
model.add(Dense(20, activation='relu', input_shape=(16,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Process
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Modified optimizer
              metrics=['accuracy'])
H = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,  # log or not
              )

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
a = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Save metrics to file
with open("result/metrics_ann.txt", "w") as f:
    f.write(f'+ precision = {a[0]:.3f}\n')
    f.write(f'+ recall = {a[1]:.3f}\n')
    f.write(f'+ f1_score = {a[2]:.3f}\n')
    f.write(f'+ accuracy = {accuracy:.3f}\n')

# Save confusion matrix plot to file
plt.figure(figsize=(8, 6))
ax = sns.heatmap(data=confusion_matrix_result, fmt="d", annot=True)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
plt.savefig("result/confusion_matrix_ann.png")  # Save the plot to a file
# plt.close()


# # Thử với các learning rate khác nhau (giữ nguyên các layer, node và activation function)
# learning_rate = [0.1, 0.005, 0.01, 0.00001, 0.2]
# colors = ['r', 'g', 'b', 'y', 'c']
# for i, lr in enumerate(learning_rate):
#     model = Sequential()
    
#     model.add(Dense(10, activation='relu', input_shape=(16,)))
#     model.add(Dense(num_classes, activation='softmax'))

#     model.compile(loss='categorical_crossentropy',
#                     optimizer=Adam(learning_rate=lr),
#                     metrics=['accuracy'])
#     print('\nLearning rate = %f' %(lr))
#     H = model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,  # log or not
#               )
#     plt.plot(H.history['accuracy'], colors[i])
  
# plt.title('Loss with Different Learning rates')
# plt.legend(learning_rate)
# plt.tight_layout()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.show()