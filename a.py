import matplotlib.pyplot as plt

sceneflow2kitti_alg_zs = {
    'CLIP-32': 0.242,
    'CLIP-16': 0.2,
    'DINO': 0.192,
    'RESNET': 0.36,
    'EFFICIENTNET': 0.35
}
sceneflow2kitti_sed_zs = {
    'CLIP-32': 0.27,
    'CLIP-16': 0.182,
    'DINO': 0.198,
    'RESNET': 0.39,
    'EFFICIENTNET': 0.37
}

kitti2sceneflow_alg_zs = {
    'CLIP-32': 0.549 ,
    'CLIP-16': 0.462,
    'DINO':0.4639,
    'RESNET':0.582,
    'EFFICIENTNET': 0.64
}
kitti2sceneflow_sed_zs = {
    'CLIP-32': 1.447,
    'CLIP-16': 1.32,
    'DINO': 1.2,
    'RESNET':1.78,
    'EFFICIENTNET': 2
}

sceneflow2kitti_alg_ft = {
    'CLIP-32': 0.21,
    'CLIP-16': 0.183,
    'DINO': 0.178,
    'RESNET': 0.294,
    'EFFICIENTNET': 0.3}
sceneflow2kitti_sed_ft = {
    'CLIP-32': 0.23,
    'CLIP-16': 0.155,
    'DINO': 0.175,
    'RESNET': 0.318,
    'EFFICIENTNET': 0.34
}

kitti2sceneflow_alg_ft = {
    'CLIP-32': 0.48,
    'CLIP-16': 0.466,
    'DINO':0.41,
    'RESNET':0.51,
    'EFFICIENTNET':0.54 
}
kitti2sceneflow_sed_ft = {
    'CLIP-32': 1.21,
    'CLIP-16': 1.12,
    'DINO':0.954,
    'RESNET':1.56,
    'EFFICIENTNET': 1.65
}

kitti_alg_32 = {
    'CLIP-32': 0.57,
    'CLIP-16': 0.364,
    'DINO': 0.39,
    'RESNET': 0.44,
    'EFFICIENTNET': 0.464
}
kitti_sed_32 = {
    'CLIP-32': 1.05,
    'CLIP-16': 0.44,
    'DINO': 0.45,
    'RESNET': 0.562,
    'EFFICIENTNET': 0.684
}

kitti_alg_2100 = {
    'CLIP-32': 0.23,
    'CLIP-16': 0.16,
    'DINO': 0.17,
    'RESNET': 0.23,
    'EFFICIENTNET': 0.24}
kitti_sed_2100 = {
    'CLIP-32': 0.199,
    'CLIP-16': 0.12,
    'DINO': 0.125,
    'RESNET': 0.21,
    'EFFICIENTNET': 0.24}

sceneflow_alg_1600 = {
    'CLIP-32': 0.293,
    'CLIP-16': 0.35,
    'DINO': 0.34,
    'RESNET': 0.4,
    'EFFICIENTNET': 0.415}
sceneflow_sed_1600 = {
    'CLIP-32': 0.54,
    'CLIP-16': 0.69,
    'DINO': 0.7,
    'RESNET': 0.885,
    'EFFICIENTNET': 0.872}

sceneflow_alg_80 = {
    'CLIP-32': 0.67,
    'CLIP-16': 0.578,
    'DINO':0.589,
    'RESNET':0.66,
    'EFFICIENTNET': 0.6
}
sceneflow_sed_80 = {
    'CLIP-32': 2.1,
    'CLIP-16': 1.74,
    'DINO':1.814,
    'RESNET':2,
    'EFFICIENTNET': 1.8
}

scene2kitti = False

# Plotting
models = list(kitti_sed_32.keys())
x = range(len(models))# Define colors for each model
colors = ['darkblue', 'blue', 'deepskyblue', "lightsteelblue", 'orange']
plt.figure(figsize=(11, 8.2))

# Custom legend for filled and hollow circles
legend_elements = [
    plt.Line2D([0], [0], marker='v', color=colors[0], markersize=9, linestyle='None', label='KITTI 2166' if scene2kitti else 'FlyingThigs 1431'),
    plt.Line2D([0], [0], marker='o', color=colors[1], markersize=9, linestyle='None', label='FlyingThings3D -> KITTI Fine-tuned' if scene2kitti else 'KITTI -> FlyingThings3D Fine-tuned'),
    plt.Line2D([0], [0], marker='o', color=colors[2], markersize=9, linestyle='None', label='FlyingThings3D -> KITTI Zero-Shot' if scene2kitti else 'KITTI -> FlyingThings3D Zero-Shot'),
    plt.Line2D([0], [0], marker='^', color=colors[3], markersize=9, linestyle='None', label='KITTI 32' if scene2kitti else 'FlyingThigs 80'),
    plt.Line2D([0], [0], marker='*', color=colors[4], markersize=9, linestyle='None', label='FlyingThigs 1431' if scene2kitti else 'KITTI 2166' )

]


# Plot for ALG metric
plt.subplot(1, 2, 1)
for i, model in enumerate(models):
    plt.scatter(i, kitti_alg_2100[model] if scene2kitti else sceneflow_alg_1600[model], color=colors[0], marker='v', s=105)
    plt.scatter(i, sceneflow2kitti_alg_ft[model] if scene2kitti else kitti2sceneflow_alg_ft[model], color=colors[1], marker='o', s=105)
    plt.scatter(i, sceneflow2kitti_alg_zs[model] if scene2kitti else kitti2sceneflow_alg_zs[model], color=colors[2], marker='o', s=105)
    plt.scatter(i, kitti_alg_32[model] if scene2kitti else sceneflow_alg_80[model], color=colors[3], marker='^', s=105)
    plt.scatter(i, sceneflow_alg_1600[model] if scene2kitti else kitti_alg_2100[model], color=colors[4], marker='*', s=105)
plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('ALG Metric', fontsize=12)
plt.title('ALG Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.05, 1.185), ncol=3, fontsize=13)
plt.subplots_adjust(wspace=0.3)  # Increase the space between subplots

# Plot for SED metric
plt.subplot(1, 2, 2)
for i, model in enumerate(models):
    plt.scatter(i, kitti_sed_2100[model] if scene2kitti else sceneflow_sed_1600[model], color=colors[0], marker='v', s=105)
    plt.scatter(i, sceneflow2kitti_sed_ft[model] if scene2kitti else kitti2sceneflow_sed_ft[model], color=colors[1], marker='o', s=105)
    plt.scatter(i, sceneflow2kitti_sed_zs[model] if scene2kitti else kitti2sceneflow_sed_zs[model], color=colors[2], marker='o', s=105)
    plt.scatter(i, kitti_sed_32[model] if scene2kitti else sceneflow_sed_80[model], color=colors[3], marker='^', s=105)
    plt.scatter(i, sceneflow_sed_1600[model] if scene2kitti else kitti_sed_2100[model], color=colors[4], marker='*', s=105)
plt.xticks(x, models, rotation=35, fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel('SED Metric', fontsize=12)
plt.title('SED Metric Comparison', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplots_adjust(wspace=0.3, bottom=0.125, top=0.815)  # Adjust bottom to prevent label cutoff
plt.suptitle('KITTI -> FlyingThings Fine tune/Zero-Shot' if not scene2kitti else 'FlyingThings -> KITTI Fine tune/Zero-Shot', fontsize=16, fontweight='bold')

plt.subplots_adjust(left=0.062, right=0.97)

plt.savefig('results/sceneflow2kitti' if scene2kitti else 'results/kitti2sceneflow')
# plt.show()