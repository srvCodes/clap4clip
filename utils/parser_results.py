import os 
import re 
from pathlib import Path 
import numpy as np 

def get_average_results(fname, method="method1"):
    with open(Path(fname), "r") as f:
        all_lines = f.readlines()
    avg_accs, last_accs, calibration_errors, joint_accuracies = [], [], [], []
    for line in all_lines:
        if line.startswith("Acc avg"):
            result = re.findall(r'(?<!\S)[0-9]\S*[,0-9](?!\S)', line)
            
            avg_accs.append(float(result[0].replace(',', ''))*100.)
            last_accs.append(float(result[1])*100.)
        if line.startswith("Avg. Expected Calibration Error"):
            result = re.findall(r'(?<!\S)[0-9]\S*[,0-9](?!\S)', line)
            calibration_errors.append(float(result[0]))

        if method == "method2":
            if line.startswith("Joint accuracy"):
                result = re.findall(r'(?<!\S)[0-9]\S*[,0-9](?!\S)', line)
                joint_accuracies.append(result)
    
    total_len = 132 if method == "method2" else 60
    if len(avg_accs) < total_len:
        if method == "method2":
            len_to_keep = (len(avg_accs) // 22) * 22
            avg_accs = avg_accs[:len_to_keep]
            last_accs = last_accs[:len_to_keep]
            calibration_errors = calibration_errors[:len_to_keep]
            total_len = len_to_keep
    assert len(avg_accs) == total_len and len(last_accs) == total_len and len(calibration_errors) == total_len
    return np.array(avg_accs), np.array(last_accs), np.array(calibration_errors)

def format_for_method_1(avg_accs, last_accs, calibration_errors):
    avg_accs = avg_accs.reshape(6, 10)
    print(f"Avg acc for ViT-B-16: {avg_accs[:3].mean(0)}, for ViT-L-14: {avg_accs[3:].mean(0)}")

    last_accs = last_accs.reshape(6, 10)
    print(f"Last acc for ViT-B-16: {last_accs[:3].mean(0)}, for ViT-L-14: {last_accs[3:].mean(0)}")

    errors = calibration_errors.reshape(6, 10)
    print(f"Calibration error for ViT-B-16: {errors[:3].mean(0)}, for ViT-L-14: {errors[3:].mean(0)}")
    
def helper_average(figures, descr="Avg acc", model="ViT-B-16", start_idx=0, end_idx=0):
    print(f"{descr} for {model}: {figures[:, start_idx:end_idx].mean(0)}")

def format_for_method_2(avg_accs, last_accs, calibration_errors):
    num_runs = len(avg_accs) // 22
    avg_accs = avg_accs.reshape(num_runs, 22)
    last_accs = last_accs.reshape(num_runs, 22)
    for (start_idx, end_idx, model) in [[0, 3, "ViT-B-16"], [3,num_runs, "ViT-L-14"]]:
        helper_average(avg_accs[start_idx:end_idx], start_idx=0, end_idx=10, model=model)
        helper_average(last_accs[start_idx:end_idx], "Last acc", end_idx=10, model=model)
        helper_average(last_accs[start_idx:end_idx], start_idx=19, end_idx=20, descr="CIFAR100-I2C", model=model)
        helper_average(last_accs[start_idx:end_idx], start_idx=20, end_idx=21, descr="ImageNet100-I2C", model=model)
        helper_average(last_accs[start_idx:end_idx], start_idx=21, end_idx=22, descr="CIFAR100+ImageNet100", model=model)
        print("\n")

    
fname = "runner_clip_adapter_imagenet.sh.o96873725"
avg_accs, last_accs, calibration_errors = get_average_results(fname, method="method2")
print(last_accs); exit(1)
format_for_method_2(avg_accs, last_accs, calibration_errors)