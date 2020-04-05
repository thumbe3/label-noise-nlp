import sys
import random
import os
import pdb
import argparse
random.seed(0)

def add_noise(orig,new,noise,num_classes):
    count=0
    changed=0
    for line in orig:
        parts=line.strip().split("\t")
        label=int(parts[1])
        if random.random() < noise:   
            # random noise
            new_label=random.randint(0,num_classes-1)
            while new_label == label:
                new_label=random.randint(0,num_classes-1)

            # (label+1) noise
            #new_label = (label+1)%num_classes
            
            #if len(parts[0].split()) > 30:
            new.write(parts[0]+"\t"+str(new_label)+"\t"+"1"+"\n")
            changed+=1
            #else:
            #    new.write(line)
        else:
            new.write(line)
        count+=1
    print(count,changed)
    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    #argparser.add_argument("--rand_noise", action='store_true', help="whether to use uniform random noise")
    argparser.add_argument("--dataset", type=str, default="ag_news", help="which dataset")
    argparser.add_argument("--num_classes", type=int, default=4)
    argparser.add_argument("--noise", type=float, default=0.5)
    args = argparser.parse_args()

    train_file = open(os.path.join("data", args.dataset,"train_orig.tsv"),"r")
    dev_file = open(os.path.join("data", args.dataset,"dev_orig.tsv"),"r")
    new_train_file = open(os.path.join("data", args.dataset,"train.tsv"),"w+")
    new_dev_file = open(os.path.join("data", args.dataset,"dev.tsv"),"w+")
    add_noise(train_file,new_train_file, args.noise, args.num_classes)
    add_noise(dev_file,new_dev_file,args.noise, args.num_classes)
