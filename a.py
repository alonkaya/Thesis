import matplotlib.pyplot as plt

kitti_alg = {
    'CLIP-32': 0.23,
    'CLIP-16': 0.16,
    'DINO': 0.17,
    'RESNET': 0.23,
    'EFFICIENTNET': 0.24}
kitti_sed = {
    'CLIP-32': 0.199,
    'CLIP-16': 0.12,
    'DINO': 0.125,
    'RESNET': 0.21,
    'EFFICIENTNET': 0.24}
sceneflow2kitti_alg = {
    'CLIP-32': 0.245,
    'CLIP-16': 0.2530944,
    'DINO': 0.265467904,
    'RESNET': 0.33183488,
    'EFFICIENTNET': 0.319461376}
sceneflow2kitti_sed = {
    'CLIP-32': 0.265,
    'CLIP-16': 0.274466816,
    'DINO': 0.282430864,
    'RESNET': 0.403826176,
    'EFFICIENTNET': 0.391452672}
sceneflow2kitti_ft_alg = {
    'CLIP-32': 0.249,
    'CLIP-16': 0.165,
    'DINO': 0.2,
    'RESNET': 0.26,
    'EFFICIENTNET': 0.28
}
sceneflow2kitti_ft_sed = {
    'CLIP-32': 0.222,
    'CLIP-16': 0.14,
    'DINO': 0.15,
    'RESNET': 0.26,
    'EFFICIENTNET': 0.265
}


# Plotting
models = list(kitti_alg.keys())
x = range(len(models))# Define colors for each model
colors = ['darkblue', 'lightblue', 'royalblue']  # Colors for each model

# Custom legend for filled and hollow circles
legend_elements = [
    plt.Line2D([0], [0], marker='o', color=colors[1], markersize=8, linestyle='None', label='FlyingThings3D -> KITTI Zero-Shot'),
    plt.Line2D([0], [0], marker='o', color=colors[2], markersize=8, linestyle='None', label='FlyingThings3D -> KITTI Fine-tuned'),
    plt.Line2D([0], [0], marker='o', color=colors[0], markersize=8, linestyle='None', label='KITTI')
]

# Improved Plot with Axis Lines
plt.figure(figsize=(11, 8.6))

# Plot for ALG metric
plt.subplot(1, 2, 1)
for i, model in enumerate(models):
    plt.scatter(i, kitti_alg[model], color=colors[0], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_alg[model], color=colors[1], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_ft_alg[model], color=colors[2], marker='o', s=95., linewidth=0.5)  # Filled circle

plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('ALG Metric', fontsize=12)
plt.title('ALG Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.2, 1.125), ncol=3, fontsize=13)
plt.subplots_adjust(wspace=0.3)  # Increase the space between subplots

# Plot for SED metric
plt.subplot(1, 2, 2)
for i, model in enumerate(models):
    plt.scatter(i, kitti_sed[model], color=colors[0], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_sed[model], color=colors[1], marker='o', s=95, linewidth=0.5)  # Filled circle
    plt.scatter(i, sceneflow2kitti_ft_sed[model], color=colors[2], marker='o', s=95, linewidth=0.5)  # Filled circle

plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('SED Metric', fontsize=12)
plt.title('SED Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)


plt.savefig('results/sceneflow2kitti')
