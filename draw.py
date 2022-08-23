import matplotlib.pyplot as plt

dataset = 'UTKFace'
task1 = 'gender'
task2 = 'white'

task1_baseline = 0.901
task2_baseline = 0.86

# LC_task1_list = [0.91, 0.9135, 0.9196, 0.9196, 0.9094]
# LC_task2_list = [0.958, 0.9598, 0.9575, 0.9605, 0.9494]
#
# EPO_task1_list = [0.8929, 0.9148, 0.9161, 0.9162, 0.9223]
# EPO_task2_list = [0.9634, 0.9582, 0.9561, 0.9301, 0.8244]

LC_task1_list = [0.878, 0.896, 0.893, 0.9, 0.8963]
LC_task2_list = [0.87, 0.864, 0.865, 0.851, 0.8436]

EPO_task1_list = [0.8066, 0.8736, 0.8856, 0.887, 0.8993]
EPO_task2_list = [0.8493, 0.8643, 0.8603, 0.839, 0.795]

plt.figure()
X1 = [0.6, 1]
Y1 = [task2_baseline, task2_baseline]
plt.plot(X1, Y1, linestyle='--', color='gray')

X2 = [task1_baseline, task1_baseline]
Y2 = [0.6, 1]
plt.plot(X2, Y2, linestyle='--', color='gray')

plt.scatter(LC_task1_list, LC_task2_list, marker='s',color='blue', label='LC')
plt.scatter(EPO_task1_list, EPO_task2_list, marker='8',color='red', label='EPO')

plt.legend()
plt.show()


