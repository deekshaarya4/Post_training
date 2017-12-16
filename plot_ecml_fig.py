import pickle
curve = []
with open('exps/final_results/classic/results.pkl', 'rb') as pickle_file:
    curve = pickle.load(pickle_file)

import matplotlib.pyplot as plt
import numpy as np

fig, (ax2, ax1) = plt.subplots(2, 1, figsize=[10.0, 6.0])

err = np.array(curve)
ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposy='clip')
ax2.set_xscale("log", nonposx='clip')
ax2.set_yscale("log", nonposy='clip')
ax1.plot(err[:, 0], err[:, 3], 'b', dashes=[6, 2], label="Regular Training")
ax1.plot(err[:, 0], err[:, 1], 'r', label="Post Training")
# ax1.set_ylim((.13, .3))
ax1.set_xlim((500, 50000))

# ax1.set_xticks(np.linspace(0, 120000, 7))
# ax1.set_yticks([.13, .15, .2, .3])
# ax1.set_yticklabels([.13, '', .2])
ax1.set_ylabel("Classification Error", fontsize="x-large")

ax2.plot(err[:, 0], err[:, 2], 'r', label="Post Training")
ax2.plot(err[:, 0], err[:, 4], 'b', dashes=[6, 2], label="Regular Training")
ax2.legend()
# ax2.set_ylim(.2, 2)
# ax2.set_xticks(np.linspace(0, 120000, 7))
# ax2.set_xticklabels(
#     ['{}k'.format(k * 20) if k > 0 else 0 for k in range(7)])
# ax2.set_yticks([.2, .5, 1])
# ax2.set_yticklabels([.2, .5, 1])
ax2.set_xlim((500, 50000))
plt.xlabel("Iteration $T$", fontsize="x-large")
ax2.set_ylabel("Training Cost", fontsize="x-large")
plt.subplots_adjust(left=.09, bottom=.1, right=.97, top=.97)
# plt.savefig(os.path.join(exp_dir, "cifar10_full_ecml.pdf"), dpi=150)
plt.show()
