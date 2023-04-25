import numpy as np
import csv

# data = np.loadtxt(open("./output.tmp.csv","rb"),delimiter=",")
with open("./output.tmp.csv", encoding='utf-8') as f:
    data = np.loadtxt("./output.tmp.csv", dtype=str, delimiter=',')

# print(data)
output_dict = {}
for i in data:
    # print(type(i), i[0], i[1])
    if i[0] in output_dict:
        value = float(i[1])
        if value < output_dict[i[0]]:
            output_dict[i[0]] = value
    else:
        output_dict[i[0]] = float(i[1])

out_d = {}
for key_ in output_dict:
    if len(key_.split('/')) > 1:
        nkey = key_.split('/')[-1]
        out_d[nkey] = str( output_dict[key_])+'ms'
    else:
        out_d[key_] = str(output_dict[key_])+'ms'

# print(out_d)

with open('output.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in out_d.items():
        writer.writerow(row)