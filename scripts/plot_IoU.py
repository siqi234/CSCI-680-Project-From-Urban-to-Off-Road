import matplotlib.pyplot as plt

# Data
rounds_labels = ['0', '1', '2', '3', '4', '5', 'FULL']
x_indices = range(len(rounds_labels))

unet_miou = [0.470748, 0.919360, 0.928127, 0.935585, 0.941459, 0.941214, 0.941035]
segformer_miou = [0.824622, 0.914916, 0.924513, 0.922340, 0.927172, 0.929413, 0.938663]
deeplabv3_miou = [0.738024, 0.912750, 0.926629, 0.929614, 0.932425, 0.927862, 0.935505]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x_indices, unet_miou, marker='o', label='Unet')
plt.plot(x_indices, segformer_miou, marker='s', label='Segformer')
plt.plot(x_indices, deeplabv3_miou, marker='^', label='Deeplabv3++')

plt.title('Comparison of Mean IoU Across Rounds', fontsize=14)
plt.xlabel('Round Number', fontsize=12)
plt.ylabel('Mean IoU', fontsize=12)
plt.xticks(x_indices, rounds_labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust y-axis to see the high values better, but keep 0 in view or zoom in?
# Given the baseline starts low for Unet, full range is good, but maybe zoom in on top part?
# Let's keep it auto but maybe set a lower limit if needed. Auto is usually fine.
# Actually, Unet starts at 0.47, others higher. The plot will show this large gap.

plt.tight_layout()
plt.savefig('plots/model_comparison.png')