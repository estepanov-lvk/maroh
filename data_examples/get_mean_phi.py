import os
from pathlib import Path
import json

directory_start = Path("/home/uberariy/dte-stand/data_examples/rhombus/")
experiments = [
"17-11-34,24-03-2024_TWOL_rhombus_memmory_size256_threshold0.01",
#
"13-15-35,25-03-2024_TWOL_rhombus_memmory_size256_threshold0.01_storelinkstates",
#
"21-37-27,25-03-2024_TWOL_rhombus_memmory_size256_threshold0.01_storelinkstates_expgamma0.35",
#
"23-15-02,27-03-2024_TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.40",
# Episode  4628, message_iterations done 3033823 / 3703200. 
"22-21-41,26-03-2024_TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.00",
#
"12-30-57,26-03-2024_TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.35",
#
"18-26-36,30-03-2024_TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.80"
]
experiment_num = 6
directory = os.fsencode(directory_start / experiments[experiment_num] / "iteration0")
print("EXP: ", experiments[experiment_num])
    
files = dict()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".json"): 
        tmp = filename[:-5].split('_')[-1]
        with open(directory_start / experiments[experiment_num] / "iteration0" / filename) as f:
            data = json.load(f)
        if filename.startswith("phi_values"):
            mea = [sum(d) / len(d) for d in data]
            mea2 = sum(mea) / len(mea)
            files[int(tmp.split('-')[0])] = (tmp, mea2)
        if filename.startswith("messages"):
            pass
        # print(filename[:-5].split('_')[-1], )

# print([k for k in files.keys()])
tmp = 0
numb = 5
for n, f in enumerate(sorted([k for k in files.keys()])):
    print(files[f][0], "\t", files[f][1], end="\t")
    if (n % numb == numb - 1):
        tmp += files[f][1]
        print(f"/ {tmp / numb}")
        tmp = 0
    else:
        tmp += files[f][1]
        print("|")