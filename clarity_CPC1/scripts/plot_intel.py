import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
with open("metadata/CPC1.train.json") as f:
    train_data = json.load(f)

c_list = []
v_list = []
n_words_list = []
system_list = []
#print(train_data)
for row in train_data:
    #print(row)
    #print(row["correctness"])
    c_list.append(row["correctness"])
    v_list.append(row["volume"])
    n_words_list.append(row["n_words"])
    system_list.append(row["system"])

haspi_list = []
with open("haspi2.train.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        #print(row)
        scene,listener,system,HASPI = row
        HASPI =ast.literal_eval(HASPI)
        #print(HASPI)
        haspi_list.append(np.mean(HASPI))
print(haspi_list[0:10])
print(c_list[0:10])
plt.scatter(c_list,haspi_list)
plt.ylabel("Mean HASPI score")
plt.xlabel("Ground Truth Correctness")
plt.savefig("scatter.png")
plt.close()


plt.scatter(c_list,v_list)
plt.ylabel("Volume")
plt.xlabel("Ground Truth Correctness")
plt.savefig("scatter_volume.png")
plt.close()

plt.scatter(n_words_list,haspi_list)
plt.ylabel("HASPI")
plt.xlabel("Number of words")
plt.savefig("scatter_words.png")
plt.close()

plt.hist(c_list)
plt.savefig("c_hist.png")
plt.close()
plt.hist(haspi_list)
plt.savefig("haspi_hist.png")
plt.close()


