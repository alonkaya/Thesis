import matplotlib.pyplot as plt

sceneflow2kitti_alg = {
    'CLIP-32': 0.242,
    'CLIP-16': 0.18,
    'RESNET': 0.282,
    'DINO': 0.194,
    'EFFICIENTNET': 0.285}
sceneflow2kitti_sed = {
    'CLIP-32': 0.222,
    'CLIP-16': 0.16,
    'RESNET': 0.295,
    'DINO': 0.165,
    'EFFICIENTNET': 0.34}
kitti_alg = {
    'CLIP-32': 0.23,
    'CLIP-16': 0.16,
    'RESNET': 0.23,
    'DINO': 0.17,
    'EFFICIENTNET': 0.24}
kitti_sed = {
    'CLIP-32': 0.199,
    'CLIP-16': 0.12,
    'RESNET': 0.21,
    'DINO': 0.125,
    'EFFICIENTNET': 0.24}


# sceneflow2kitti_ft_alg = {
#     'CLIP-32': 0.249,
#     'CLIP-16': 0.165,
#     'DINO': 0.2,
#     'RESNET': 0.26,
#     'EFFICIENTNET': 0.28
# }
# sceneflow2kitti_ft_sed = {
#     'CLIP-32': 0.222,
#     'CLIP-16': 0.14,
#     'DINO': 0.15,
#     'RESNET': 0.26,
#     'EFFICIENTNET': 0.265
# }


# Plotting
models = list(kitti_alg.keys())
x = range(len(models))# Define colors for each model
colors = ['darkblue', 'lightblue', 'royalblue']  # Colors for each model

# Custom legend for filled and hollow circles
legend_elements = [
    plt.Line2D([0], [0], marker='o', color=colors[1], markersize=8, linestyle='None', label='FlyingThings3D -> KITTI Zero-Shot'),
    # plt.Line2D([0], [0], marker='o', color=colors[2], markersize=8, linestyle='None', label='FlyingThings3D -> KITTI Fine-tuned'),
    plt.Line2D([0], [0], marker='o', color=colors[0], markersize=8, linestyle='None', label='KITTI')
]

# Improved Plot with Axis Lines
plt.figure(figsize=(11, 6.2))

# Plot for ALG metric
plt.subplot(1, 2, 1)
for i, model in enumerate(models):
    plt.scatter(i, kitti_alg[model], color=colors[0], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_alg[model], color=colors[1], marker='o', s=95, linewidth=0.5)  # Filled circle
    # plt.scatter(i, sceneflow2kitti_ft_alg[model], color=colors[2], marker='o', s=95., linewidth=0.5)  # Filled circle

plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('ALG Metric', fontsize=12)
plt.title('ALG Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.4, 1.17), ncol=2, fontsize=13)
plt.subplots_adjust(wspace=0.3)  # Increase the space between subplots

# Plot for SED metric
plt.subplot(1, 2, 2)
for i, model in enumerate(models):
    plt.scatter(i, kitti_sed[model], color=colors[0], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_sed[model], color=colors[1], marker='o', s=95, linewidth=0.5)  # Filled circle
    # plt.scatter(i, sceneflow2kitti_ft_sed[model], color=colors[2], marker='o', s=95, linewidth=0.5)  # Filled circle

plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('SED Metric', fontsize=12)
plt.title('SED Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplots_adjust(wspace=0.4, bottom=0.16)  # Adjust bottom to prevent label cutoff


plt.savefig('results/sceneflow2kitti')
plt.show()