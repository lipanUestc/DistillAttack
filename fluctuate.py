import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

subtitle_font = FontProperties(size=22)
# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    poison_values = []
    acc_values = []
    for line in content:
        parts = line.split(',')
        poison = float(parts[0].split(':')[1].strip())
        acc = float(parts[1].split(':')[1].strip())
        poison_values.append(poison)
        acc_values.append(acc)
    return poison_values, acc_values

# Function to calculate rolling standard deviation
def rolling_std(values, window):
    return np.array([np.std(values[max(0, i-window):i+1]) for i in range(len(values))])

# Load data from the original file
original_file_path = 'MGDA_WSR_Acc.txt'  # Replace with the actual path
poison_values, acc_values = load_data(original_file_path)

# Load data from the new file and replace the poison values
new_file_path = 'MGDA_cifar10.txt'  # Replace with the actual path
new_poison_values, _ = load_data(new_file_path)
poison_values = new_poison_values

# Calculate rolling standard deviation
window_size = 5  # Example window size, adjust as needed
rolling_std_wsr = rolling_std(poison_values, window_size)
rolling_std_acc = rolling_std(acc_values, window_size)

# Plotting the data
plt.figure(figsize=(8, 6))

plt.plot(range(len(poison_values)), poison_values, label='WSR', color='orange', linestyle='-')
plt.fill_between(range(len(poison_values)), np.array(poison_values) - rolling_std_wsr, np.array(poison_values) + rolling_std_wsr, color='orange', alpha=0.3)

plt.plot(range(len(acc_values)), acc_values, label='Acc', color='green', linestyle='-')
plt.fill_between(range(len(acc_values)), np.array(acc_values) - rolling_std_acc, np.array(acc_values) + rolling_std_acc, color='green', alpha=0.3)



plt.xlabel('Epoch', fontproperties=subtitle_font)
plt.ylabel('Performance (%)', fontproperties=subtitle_font)
plt.legend()
plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)

plt.savefig('fluct.png')
plt.show()

