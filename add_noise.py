import sys
import random
import os

random.seed(0)

def add_noise(orig,new,noise,num_classes):
    count=0
    changed=0
    for line in orig:
        parts=line.strip().split("\t")
        label=int(parts[1])
        if random.random() < noise:
            new_label=random.randint(0,num_classes-1)
            while new_label == label:
                new_label=random.randint(0,num_classes-1)
            new.write(parts[0]+"\t"+str(new_label)+"\t"+"1"+"\n")
            changed+=1
        else:
            new.write(line)
        count+=1
    print(count,changed)
    return

if __name__ == "__main__":
    dataset=sys.argv[1]
    num_classes=int(sys.argv[2])
    noise=float(sys.argv[3])
    train_file = open(os.path.join("data",dataset,"train_orig.tsv"),"r")
    dev_file = open(os.path.join("data",dataset,"dev_orig.tsv"),"r")
    new_train_file = open(os.path.join("data",dataset,"train.tsv"),"w+")
    new_dev_file = open(os.path.join("data",dataset,"dev.tsv"),"w+")
    add_noise(train_file,new_train_file,noise,num_classes)
    add_noise(dev_file,new_dev_file,noise,num_classes)
