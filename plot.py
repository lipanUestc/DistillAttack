# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# Path to your TXT file
file_path = "/home/lpz/MHL/DistillAttack/plots/plot123.txt"
subtitle_font = FontProperties(size=22)

# Read the data from the TXT file
df = pd.read_csv(file_path, sep='[:,]', engine='python', header=None, skipinitialspace=True, nrows=1500)

df.columns = ['Metric', 'WSR', 'Metric2', 'Acc', 'Metric3', 'Angle']
df.drop(['Metric', 'Metric2', 'Metric3'], axis=1, inplace=True)

df = df.iloc[::5, :]


# Plotting
fig, ax1 = plt.subplots(figsize=(8, 6))

angle_at_300 = df.loc[300, 'Angle'] if 300 in df.index else "N/A"
angle_at_1000 = df.loc[1000, 'Angle'] if 1000 in df.index else "N/A"

plt.axvline(x=300, color='gray', linestyle='-', linewidth=1, zorder=0)

plt.axvline(x=1000, color='gray', linestyle='-', linewidth=1, zorder=0)

# Plotting WSR and Acc on the first y-axis without markers
ax1.plot(df.index, df['WSR'], label='WSR', color='orange')
ax1.plot(df.index, df['Acc'], label='Acc', color='green')

wsr_poly_coeffs = np.polyfit(df.index, df['WSR'], 6)
acc_poly_coeffs = np.polyfit(df.index, df['Acc'], 6)

wsr_poly = np.poly1d(wsr_poly_coeffs)
acc_poly = np.poly1d(acc_poly_coeffs)

ax1.plot(df.index, wsr_poly(df.index), linestyle='--', color='orange', linewidth=2)
ax1.plot(df.index, acc_poly(df.index), linestyle='--', color='green', linewidth=2)

ax1.set_xlabel('Steps', fontproperties=subtitle_font)
ax1.set_ylabel('WSR/Acc (%)', fontproperties=subtitle_font)
ax1.legend(loc='upper left')

plt.legend(prop={'size': 15})
plt.tick_params(axis='both', which='major', labelsize=15)

# Title and grid lines off
ax1.grid(False)
plt.tight_layout()

plt.savefig('plot2.png')
plt.show()


