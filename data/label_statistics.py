import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

f = pd.read_csv('full_data.csv')

def label_statistics():
    label_array = np.array(f["label"])
    counts = np.bincount(label_array, minlength=5)
    plt.bar(range(5), counts)
    plt.title('Biểu đồ số lượng mẫu theo giá trị')
    plt.xlabel('Giá trị')
    plt.ylabel('Số lượng mẫu')
    
    # Thêm giá trị của từng cột vào biểu đồ
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig("result.png")  # Lưu biểu đồ vào file
    plt.close()

label_statistics()
