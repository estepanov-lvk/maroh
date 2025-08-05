import sys
import os
import time

if __name__ == "__main__":
    # Change file pathes below
    path_to_main = "main.py"
    path_to_data_example = "data_examples/rhombus"
    # Change parameters grid below
    experiment_tag = "gat_otladka_reluingat_heads2"
    duration = 60
    experiment_number = 3
    t_start = time.time()
    expN = 0
    expC = experiment_number
    for i in range(experiment_number):
                expN += 1
                t_one_start = time.time()
                print(f"#### RunAll: Running experiment ({expN}/{expC}) - experiment_tag: {experiment_tag}\t... ")
                os.system(f"python3 {path_to_main} {path_to_data_example} -n {experiment_tag}")
                t_one_end = time.time()
                print(f"#### RunAll: Experiment finished. Time taken: {t_one_end - t_one_start} sec. ({(t_one_end - t_one_start) / 60} minutes)")
    t_end = time.time()
    print(f"RunAll: Total time taken: {t_end - t_start} sec. ({(t_end - t_start) / 60} minutes)")
