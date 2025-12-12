import matplotlib.pyplot as plt
import pandas as pd

# Data from previous turns
data = {
    'Model': ['Unet', 'Segformer', 'Deeplabv3++'],
    'Baseline (Round 0)': [0.470748, 0.824622, 0.738024],
    'Max Performance': [0.941459, 0.938663, 0.935505] # Max of all rounds including Full
}

df = pd.DataFrame(data)

# Calculate Absolute Gain (in percentage points)
# Formula: (Final - Initial) * 100
df['Absolute Gain (%)'] = (df['Max Performance'] - df['Baseline (Round 0)']) * 100

# Calculate Relative Improvement (%)
# Formula: ((Final - Initial) / Initial) * 100
df['Relative Improvement (%)'] = ((df['Max Performance'] - df['Baseline (Round 0)']) / df['Baseline (Round 0)']) * 100

# Plotting Absolute Gain
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['Absolute Gain (%)'], color=['blue', 'orange', 'green'], alpha=0.7)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'+{height:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Absolute mIoU Gain (Baseline vs. Best)', fontsize=14)
plt.ylabel('Gain in mIoU (%)', fontsize=12)
plt.ylim(0, 60) # Unet is ~47%, give it some headroom
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plots/iou_gain_percentage.png')

print(df[['Model', 'Absolute Gain (%)', 'Relative Improvement (%)']])