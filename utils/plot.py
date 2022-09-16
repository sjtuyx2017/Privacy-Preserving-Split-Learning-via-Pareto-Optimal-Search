import matplotlib.pyplot as plt
import matplotlib.patches as patches

test_acc_cora = [0.6962, 0.7088, 0.6811, 0.7640, 0.7265, 0.7673, 0.7260]
train_time_cora = [10, 40, 41, 402, 15, 42, 49]

labels = ['RGCN', 'GCN-Jaccard', 'GCN-SVD', 'Pro-GNN', 'SimP-GCN', 'LRGNN', 'LRGNN(S)']
colors = ['#D00000', '#FF932E', '#007A14', '#618DFF', '#D886FF', '#0094FF', '#FF3C8E']
markers = ['^', 'p', 'o', 'd', 'v', '*', 's']

plt.figure(figsize=(10,8))

for i in range(len(test_acc_cora)):
    plt.scatter(train_time_cora[i], test_acc_cora[i], c=colors[i], marker=markers[i], s=600, label=labels[i])

plt.xscale('log')
plt.xlabel('Training Time/s', fontsize=30)
plt.ylabel('Test Accuracy', fontsize=30)
plt.xticks([10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640],fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(which='major',length=7)
plt.tick_params(which='major',width=2)
plt.tick_params(which='minor',length=7)
plt.tick_params(which='minor',width=2)

ax=plt.gca()

kw = dict(arrowstyle="-|>,head_width=10,head_length=20", linestyle='--', linewidth=3, color="gray", connectionstyle="arc3")
plt.gca().add_patch(patches.FancyArrowPatch((640,0.68), (10,0.77), **kw))


ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_visible(False)

plt.legend(loc='lower right', ncol=1, fontsize=16, labelspacing=1, borderpad=1, edgecolor='black')
plt.grid(linestyle = '--', linewidth = 0.6)

plt.show()
plt.savefig('./train_cora.pdf', bbox_inches='tight')
