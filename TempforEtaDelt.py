# Temp file for plotting eta and delta values

import numpy as np
import matplotlib.pyplot as plt
eta = np.array(eta)
delta = np.array(delta)

mylabseta = ['Eta 1','Eta 2','Eta 3','Eta 4','Eta 5']
mylabsdelt = ['Delta 1','Delta 2','Delta 3','Delta 4','Delta 5']

for j in range(5):
    plt.figure(pngcount)
    plt.plot(eta[:,j],linewidth=2.0,color=mycolors[j],label=mylabseta[j])

plt.ylabel('Eta Value')
plt.xlabel('Iteration')
savestring = figsave + 'EtaLayer1_' + str(i) + '.png'
titlestring = 'Example Eta Values'
plt.title(titlestring)
plt.legend(loc=1)

plt.savefig(savestring)
plt.close()
pngcount+=1


for j in range(5):
    plt.figure(pngcount)
    plt.plot(delta[:,j],linewidth=2.0,color=mycolors[j],label=mylabseta[j])

plt.ylabel('Delta Value')
plt.xlabel('Iteration')
savestring = figsave + 'DeltaLayer1_' + str(i) + '.png'
titlestring = 'Example Delta Values'
plt.title(titlestring)
plt.legend(loc=1)

plt.savefig(savestring)
plt.close()
pngcount+=1