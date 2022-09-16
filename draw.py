import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

dataset = 'celebA'
utility = 'HighCheekbones'
privacy = 'Male'

path = './new_draw_results'

acc_baseline = [0.8799]
DSA_baseline = [0.8761]
blackbox_baseline = [0.9329]

# Noisy results
Noisy_acc_list = [0.8767,
0.8757,
0.8723,
0.8682]
Noisy_DSA_list = [0.8279,
0.7308,
0.6028,
0.5429]
Noisy_blackbox_list = [0.9302,
0.9111,
0.8447,
0.7279]

# ADV results
ADV_acc_list = [0.8737,
0.8728,
0.8538,
0.8407,
0.8314]
ADV_DSA_list = [0.6318,
0.6048,
0.6274,
0.6562,
0.6126]
ADV_blackbox_list = [0.9133,
0.9061,
0.9015,
0.895,
0.8539]

# ADV-flip results
ADV_flip_acc_list = [0.8657,
0.8472,
0.7976,
0.6433]
ADV_flip_DSA_list = [0.6201,
0.6091,
0.6159,
0.625]
ADV_flip_blackbox_list = [0.881,
0.8853,
0.8311,
0.8006]

# GMM-LC results
GMM_LC_acc_list = [0.8624,
0.8591,
0.8686,
0.8702,
0.8722,
0.8713]
GMM_LC_DSA_list = [0.5407,
0.5623,
0.5861,
0.5617,
0.5787,
0.5812]
GMM_LC_blackbox_list = [0.6127,
0.6137,
0.6419,
0.6837,
0.7349,
0.7896]

# GMM-EPO results
GMM_EPO_acc_list = [0.8644,
0.8693,
0.8728,
0.8735,
0.8712,
0.8689]
GMM_EPO_DSA_list = [0.5584,
0.5973,
0.5617,
0.5715,
0.6002,
0.5887]
GMM_EPO_blackbox_list = [0.6462,
0.7025,
0.7366,
0.7838,
0.6937,
0.6733]

#labels = ['w/o defense', 'Noisy', 'ADV', 'ADV-flip', 'GMM-LC', 'GMM-EPO']
#colors = ['black', '#FF932E', '#007A14', '#618DFF', '#D886FF', '#0094FF']
#markers = ['.', 'p', 'o', 'd', '^', '*']


# DSA
plt.figure(figsize=(12,8))

# for i in range(len(test_acc_cora)):
#     plt.scatter(train_time_cora[i], test_acc_cora[i], c=colors[i], marker=markers[i], s=600, label=labels[i])
plt.scatter(acc_baseline, DSA_baseline, marker='o',color='black', label='w/o defense', s=600)
plt.scatter(Noisy_acc_list, Noisy_DSA_list, marker='p',color='#FF932E', label='Noisy', s=600)
plt.scatter(ADV_acc_list, ADV_DSA_list, marker='v',color='#007A14', label='ADV', s=600)
plt.scatter(ADV_flip_acc_list, ADV_flip_DSA_list, marker='d',color='#D886FF', label='ADV-flip', s=600)
plt.scatter(GMM_LC_acc_list, GMM_LC_DSA_list, marker='^',color='#D00000', label='GMM-LC', s=600)
plt.scatter(GMM_EPO_acc_list, GMM_EPO_DSA_list, marker='*',color='#0094FF', label='GMM-EPO', s=600)


#plt.xscale('log')
plt.xlabel('Accuracy(Higher is better)', fontsize=25)
plt.ylabel('Attack success(Lower is better)', fontsize=25)
save_name1 = 'Utility=%s_Privacy=%s_Attack=DSA'%(utility, privacy)
name1 = '%s Classification Acc. under DSA to %s'%(utility, privacy)
plt.title(name1, fontsize=25)
#plt.xticks([10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640],fontsize=20)
plt.xticks(fontsize=20)
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

plt.legend(ncol=1, fontsize=16, labelspacing=1, borderpad=1, edgecolor='black')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.savefig(os.path.join(path, save_name1))
plt.show()
#plt.savefig('./train_cora.pdf', bbox_inches='tight')



# Black-Box
plt.figure(figsize=(12,8))

# for i in range(len(test_acc_cora)):
#     plt.scatter(train_time_cora[i], test_acc_cora[i], c=colors[i], marker=markers[i], s=600, label=labels[i])
plt.scatter(acc_baseline, blackbox_baseline, marker='o',color='black', label='w/o defense', s=600)
plt.scatter(Noisy_acc_list, Noisy_blackbox_list, marker='p',color='#FF932E', label='Noisy', s=600)
plt.scatter(ADV_acc_list, ADV_blackbox_list, marker='v',color='#007A14', label='ADV', s=600)
plt.scatter(ADV_flip_acc_list, ADV_flip_blackbox_list, marker='d',color='#D886FF', label='ADV-flip', s=600)
plt.scatter(GMM_LC_acc_list, GMM_LC_blackbox_list, marker='^',color='#D00000', label='GMM-LC', s=600)
plt.scatter(GMM_EPO_acc_list, GMM_EPO_blackbox_list, marker='*',color='#0094FF', label='GMM-EPO', s=600)


#plt.xscale('log')
plt.xlabel('Accuracy(Higher is better)', fontsize=25)
plt.ylabel('Attack success(Lower is better)', fontsize=25)
save_name2 = 'Utility=%s_Privacy=%s_Attack=Black-Box'%(utility, privacy)
name2 = '%s Classification Acc. under Black-Box to %s'%(utility, privacy)
plt.title(name2, fontsize=25)
#plt.xticks([10, 20, 40, 80, 160, 320, 640], [10, 20, 40, 80, 160, 320, 640],fontsize=20)
plt.xticks(fontsize=20)
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

plt.legend(ncol=1, fontsize=16, labelspacing=1, borderpad=1, edgecolor='black')
plt.grid(linestyle = '--', linewidth = 0.6)
plt.savefig(os.path.join(path, save_name2))
plt.show()
#plt.savefig('./train_cora.pdf', bbox_inches='tight')

