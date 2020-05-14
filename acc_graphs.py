import os
import re
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE



def plot_graphs(noise, result_dir, dataset, ax):
    result_baseline_dir =result_dir +"_baseline"
    result_path = os.path.join(os.getcwd(), result_dir, dataset)
    result_baseline_path = os.path.join(os.getcwd(), result_baseline_dir, dataset)
    process = Popen(["python", "best_params.py", result_path], stdout=PIPE)
    (output, err) = process.communicate()


    # Find best beta nd warmup for this noise

    for line in output.decode("utf-8").split("\n"):
        if f"Noise = {noise}" in line and "Hard case" in line:
            hard_beta = float(re.findall(r".*Beta: (\d+\.\d+)", line)[0])
            hard_warmup = int(re.findall(r".*Warmup: (\d+)", line)[0])
        elif f"Noise = {noise}" in line and "Soft case" in line:
            soft_beta = float(re.findall(r".*Beta: (\d+\.\d+)", line)[0])
            soft_warmup = int(re.findall(r".*Warmup: (\d+)", line)[0])



    global last_acc_dict
    last_acc_dict = {"baseline":[], "soft":[], "hard": []}


    def save_best_last(f,type):

        for line in f.readlines():
            if "Latest Model" in line:
                last_acc = float(re.findall(r".*Latest Model Test Acc.: (\d+\.\d+)", line)[0])
                last_acc_dict[type].append(last_acc)



    # basleine accuracies
    for i in os.listdir(result_baseline_path):
            if not os.path.isfile(os.path.join(result_baseline_path, i)) and f'results_{noise}' in i:
                f = open(os.path.join(result_baseline_path, i, 'output.txt'), "r")
                save_best_last(f, "baseline")
                f.close()

    #sodt and hard accuracies
    for i in os.listdir(result_path):
            print(f'results_{noise}_{soft_beta}_0_{soft_warmup}' )
            if not os.path.isfile(os.path.join(result_path, i)) and f'results_{noise}_{soft_beta}_0_{soft_warmup}' in i:
                f = open(os.path.join(result_path, i, 'output.txt'), "r")
                save_best_last(f, "soft")
                f.close()
            elif (not os.path.isfile(os.path.join(result_path, i))) and f'results_{noise}_{hard_beta}_1_{hard_warmup}' in i:
                f = open(os.path.join(result_path, i, 'output.txt'), "r")
                save_best_last(f, "hard")
                f.close()

    types = ["baseline", "soft", "hard"]
    print(last_acc_dict['soft'])
    #fig = plt.figure(num=None, figsize=(14, 6), dpi=80)


    for type in types:
        ax.plot(last_acc_dict[type], label=type)
    ax.legend()
    ax.set_title(f"{noise*100}% noise")
    ax.set_xlabel( "Epoch")
    ax.set_ylabel("Accuracy")






    """
    cnt=1
    for type in types:
        ax=fig.add_subplot(1, 3, cnt)
        ax.plot(last_acc_dict[type])
        plt.title(type)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim((0.4,1))
        cnt+=1
    """



noises = [0.2, 0.3, 0.4]
result_dir = "results_random"
dataset = "trec"

cnt=0
fig, axs = plt.subplots(1, 3, figsize=(14, 6))
for noise in noises:
    plot_graphs(noise, result_dir, dataset,axs[cnt])
    cnt+=1

fig.suptitle("Accuracy at various noise levels for Random noise")
fname = os.path.join("accuracy_graphs", f'Testing_Accuracy.png')
print(fname)
plt.savefig(fname)
plt.clf()