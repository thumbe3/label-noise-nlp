import sys
import random
import os
import pdb
import argparse

random.seed(0)

length_noise = {0:1, 0.1: 0.4, 0.2:0.34, 0.3:0.295, 0.4:0.27, 0.5:0.24}

def contain_how_what(line):
    part=line.strip().split("\t")[0]
    return part[:3] == 'How' or part[:4] == 'What'

def contain_reuter(line):
    part=line.strip().split("\t")[0]
    return "Reuter" in part or "reuter" in part


mode2_strings = {2: ['Reuter', 'reuter'], 3: ['AP ', 'ap '], 4: ['Reuter', 'reuter', 'AP ', 'ap ']}

def add_noise(orig,new,noise,num_classes,dataset, mode):
    #mode_names = ['Random noise', 'Label dependent noise', 'Instance specific noise', 'Length dependent'] # trec
    mode_names = ['Random noise', 'Label dependent noise', 'Reuter noise', 'AP noise', 'Reuter and ap noise'] # ag_news
    print(mode_names[mode])
    max_len = max([len(line.strip().split("\t")[0]) for line in orig])
    orig.seek(0)


    ### trec modes
    num_mode_2 = len(list(filter(lambda x: contain_how_what(x),orig)))
    orig.seek(0)
    data_len = len(orig.readlines())
    orig.seek(0)


    if dataset=='ag_news':
        count=0
        changed=0
        for line in orig:
            parts=line.strip().split("\t")
            label=int(parts[1])
            if mode==0:

                if random.random() < noise:
                    # random noise
                    new_label=random.randint(0,num_classes-1)
                    while new_label == label:
                        new_label=random.randint(0,num_classes-1)

                    new.write(parts[0]+"\t"+str(new_label)+"\t1\t"+str(label)+"\n")
                    changed+=1

                else:
                    new.write(line[:-1]+"\t%s\n"%str(label))


            elif mode==1:
                if random.random() < noise:  # circular noise
                        # random noise
                        new_label = (label+1)%num_classes

                        new.write(parts[0]+"\t"+str(new_label)+"\t1\t"+str(label)+"\n")
                        changed+=1
                else:
                    new.write(line[:-1]+"\t%s\n"%str(label))

            elif mode >= 2:
                flag = False
                word_list = mode2_strings[mode]
                for word in word_list:
                    if word in parts[0]:
                        new_label = random.randint(0,num_classes-1)
                        while new_label == label:
                            new_label=random.randint(0,num_classes-1)

                        new.write(parts[0]+"\t"+str(new_label)+"\t1\t"+str(label)+"\n")
                        changed+=1
                        flag = True
                        break

                if not flag:
                    new.write(line[:-1]+"\t%s\n"%str(label))




            count += 1
        print(count,changed)

    elif dataset == 'trec':
        count = 0
        changed = 0
        for line in orig:
            parts = line.strip().split("\t")
            label = int(parts[1])
            count += 1

            if mode == 0:  # random noise
                # print('Random noise')
                if random.random() < noise:
                    # random noise
                    new_label = random.randint(0, num_classes - 1)
                    while new_label == label:
                        new_label = random.randint(0, num_classes - 1)

                    new.write(parts[0] + "\t" + str(new_label) + "\t1\t" + str(label) + "\n")
                    changed += 1
                else:
                    new.write(line[:-1] + "\t%s\n" % str(label))

            elif mode == 1:  # only label dependent

                #                 if label<3 and random.random() < noise:
                #                         # random noise
                #                         new_label = random.randint(0,num_classes-1)
                #                         while new_label == label:
                #                             new_label=random.randint(0,num_classes-1)

                #                         new.write(parts[0]+"\t"+str(new_label)+"\t1\t"+str(label)+"\n")
                #                         changed+=1

                if random.random() < noise:  # circular noise
                    # random noise
                    new_label = (label + 1) % num_classes

                    new.write(parts[0] + "\t" + str(new_label) + "\t1\t" + str(label) + "\n")
                    changed += 1
                else:
                    new.write(line[:-1] + "\t%s\n" % str(label))


            elif mode == 2:  # instance specific
                #                 print('Instance specific noise')
                if (parts[0][:3] == 'How' or parts[0][:4] == 'What') and random.random() < noise*(data_len)/(num_mode_2):
                    new_label = random.randint(0, num_classes - 1)
                    while new_label == label:
                        new_label = random.randint(0, num_classes - 1)

                    new.write(parts[0] + "\t" + str(new_label) + "\t1\t" + str(label) + "\n")
                    changed += 1
                else:
                    new.write(line[:-1] + "\t%s\n" % str(label))

            elif mode == 3:
                if len(parts[0])>max_len*length_noise[noise]:
                    new_label = random.randint(0, num_classes - 1)
                    while new_label == label:
                        new_label = random.randint(0, num_classes - 1)

                    new.write(parts[0] + "\t" + str(new_label) + "\t1\t" + str(label) + "\n")
                    changed += 1
                else:
                    new.write(line[:-1] + "\t%s\n" % str(label))


        print(count, changed, changed / count)





if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    #argparser.add_argument("--rand_noise", action='store_true', help="whether to use uniform random noise")
    argparser.add_argument("--dataset", type=str, default="ag_news", help="which dataset")
    argparser.add_argument("--num_classes", type=int, default=4)
    argparser.add_argument("--noise", type=float, default=0.5)
    argparser.add_argument("--mode", type=int, default=0)
    args = argparser.parse_args()
    
    assert args.noise<=1.0, "Noise must be between 0 and 1"

    train_file = open(os.path.join("data", args.dataset,"train_orig.tsv"),"r")
    dev_file = open(os.path.join("data", args.dataset,"dev_orig.tsv"),"r")
    new_train_file = open(os.path.join("data", args.dataset,"train.tsv"),"w+")
    new_dev_file = open(os.path.join("data", args.dataset,"dev.tsv"),"w+")
    add_noise(train_file,new_train_file, args.noise, args.num_classes,args.dataset, args.mode)
    add_noise(dev_file,new_dev_file,args.noise, args.num_classes,args.dataset, args.mode)
