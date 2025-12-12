import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
rounds = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']
unet_miou = [0.470748, 0.919360, 0.928127, 0.935585, 0.941459, 0.941214]
segformer_miou = [0.824622, 0.914916, 0.924513, 0.922340, 0.927172, 0.929413]
deeplabv3_miou = [0.738024, 0.912750, 0.926629, 0.929614, 0.932425, 0.927862]

# Calculate Incremental Gains (Round N - Round N-1)
# We start from R1 (vs R0)
labels = ['R0->R1', 'R1->R2', 'R2->R3', 'R3->R4', 'R4->R5']
unet_gains = np.diff(unet_miou) * 100
segformer_gains = np.diff(segformer_miou) * 100
deeplabv3_gains = np.diff(deeplabv3_miou) * 100

# Create DataFrame for display
df_gains = pd.DataFrame({
    'Transition': labels,
    'Unet Gain (%)': unet_gains,
    'Segformer Gain (%)': segformer_gains,
    'Deeplabv3++ Gain (%)': deeplabv3_gains
})

# Plotting Incremental Gains
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, unet_gains, width, label='Unet', color='blue', alpha=0.8)
rects2 = ax.bar(x, segformer_gains, width, label='Segformer', color='orange', alpha=0.8)
rects3 = ax.bar(x + width, deeplabv3_gains, width, label='Deeplabv3++', color='green', alpha=0.8)

ax.set_ylabel('Incremental mIoU Gain (% points)')
ax.set_title('Incremental mIoU Gain per Round')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('plots/incremental_gains.png')

# Calculate Cumulative Gains (Round N - Round 0)
# For rounds 1 to 5
rounds_cum = ['R1', 'R2', 'R3', 'R4', 'R5']
unet_cum = [(x - unet_miou[0])*100 for x in unet_miou[1:]]
segformer_cum = [(x - segformer_miou[0])*100 for x in segformer_miou[1:]]
deeplabv3_cum = [(x - deeplabv3_miou[0])*100 for x in deeplabv3_miou[1:]]

df_cum = pd.DataFrame({
    'Round': rounds_cum,
    'Unet Cumulative Gain (%)': unet_cum,
    'Segformer Cumulative Gain (%)': segformer_cum,
    'Deeplabv3++ Cumulative Gain (%)': deeplabv3_cum
})

print("Incremental Gains:")
print(df_gains)
print("\nCumulative Gains:")
print(df_cum)