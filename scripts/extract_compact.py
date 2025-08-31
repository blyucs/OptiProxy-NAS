import itertools
import re
import numpy as np
with open('./p201-0/cifar10/predictors/gcn/search/20240523_091404_c10/20240523-091424-8/log.log','r') as l:
    lines = l.readlines()
    compacts = []
    accs = []
    for ll in lines:
        compact = re.findall('\[(.*?)\]', ll)
        acc = re.findall('acc = (.*?),', ll)
        if len(compact) > 1:
            compacts.append(compact[1])
            accs.append(float(acc[0]))
m = 0
mcompacts = []
count = 0
for i in range(0,len(accs)):
    if count == 5:
        count = 0
        mcompacts.append(compacts[mi])
        m = 0
    if accs[i] > m:
        m = accs[i]
        mi = i
    count += 1

mcompacts.append(compacts[-1])

X = list(itertools.product(range(5), repeat=6))
indices = []
for compact in mcompacts:
    indices.append(X.index(tuple(np.fromstring(compact, dtype=int, sep=', '))))
np.save('troj.npy', np.array(indices))