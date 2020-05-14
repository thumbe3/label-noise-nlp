import os
import re
import sys

if __name__ == "__main__":
    noises = [0.1,0.2,0.3,0.4,0.5]
    noises = [2,3,4]
    if (len(sys.argv)>1):
        result_folder=sys.argv[1]
    else:
        result_folder="cnn/results_instance_specific/ag_news"

    path = os.path.join(os.getcwd(), result_folder)
    print(path)

    folders = {}

    for i in os.listdir(path):
        for noise in noises:
            if not os.path.isfile(os.path.join(path, i)) and f'results_{noise}' in i:
                if noise not in folders:
                    folders[noise]=[]
                folders[noise].append(i)

    hard_beta = None
    hard_warmup = None
    soft_beta = None
    soft_warmup = None


    for noise in noises:
        hard_best_acc = 0
        hard_last_acc = 0
        soft_best_acc = 0
        soft_last_acc = 0

        for folder in folders[noise]:
            f = open(os.path.join(path,folder,'output.txt'), "r")
            c_best = 0
            c_last = 0
            rounding = folder.split("_")[3]

            # find best and last of the file
            for line in reversed(f.readlines()):
                if "Best Model" in line:
                    best_acc = float(re.findall(r".*Best Model Test Acc.: (\d+\.\d+)", line)[0])
                    c_best = 1
                if "Latest Model" in line:
                    last_acc = float(re.findall(r".*Latest Model Test Acc.: (\d+\.\d+)", line)[0])
                    c_last = 1
                if c_best >= 1 and c_last >= 1:
                    break
            if c_best<1 or c_last<1:
                continue
            #update hard acc
            if int(rounding) == 1 and (best_acc > hard_best_acc or (best_acc == hard_best_acc and last_acc >= hard_last_acc)):
                hard_best_acc = best_acc
                hard_last_acc = last_acc
                hard_beta, _, hard_warmup = folder.split("_")[2:]

            # update soft acc
            if int(rounding) == 0 and (best_acc > soft_best_acc or (best_acc == soft_best_acc and last_acc >= soft_last_acc)):
                soft_best_acc = best_acc
                soft_last_acc = last_acc
                soft_beta, _, soft_warmup = folder.split("_")[2:]

        print(f"Noise = {noise}, Hard case, Beta: {hard_beta} and  Warmup: {hard_warmup}")
        print(f"Best Accuracy: {hard_best_acc}, Last accuracy: {hard_last_acc}")
        print("---------------------------------------------------------------------")
        #print(f"{hard_best_acc*100} {hard_last_acc*100}")


        print(f"Noise = {noise}, Soft case, Beta: {soft_beta} and Warmup: {soft_warmup}")
        print(f"Best Accuracy: {soft_best_acc}, Last accuracy: {soft_last_acc}")
        print("---------------------------------------------------------------------")
        #print(f"{soft_best_acc*100} {soft_last_acc*100}")


