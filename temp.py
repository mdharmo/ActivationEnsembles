# Here is where I make a table

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

savestring = '/home/mharmon/ZProject/ModelResults/'+ 'TableResults.png'
cell_text = [[98.53, 97.58],[99.04,98.86],[96.3,94.2]]
rows = ['MNIST Feed','MNIST Conv','ISOLET Feed']
cols = ['Z', 'Regular']
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=cols, loc=0)
plt.savefig(savestring)
plt.close()
