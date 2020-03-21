import sys
import argparse
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

import dataloader
import modules
import os
import scipy.stats as stats
import math

from PIL import Image

def track_training_loss(model, noise_model, train_x, train_y, bmm_model, epoch):
    model.eval()
    noise_model.eval()
    with torch.no_grad():
        all_losses = torch.Tensor()

        for x, y in zip(train_x, train_y):
            data, target = Variable(x), Variable(y)
            prediction = noise_model(model(data))

            prediction = F.log_softmax(prediction, dim=1)
            idx_loss = F.nll_loss(prediction, target, reduction = 'none').detach_()
            all_losses = torch.cat((all_losses, idx_loss.cpu()))
            
    model.train()
    noise_model.train()
    
    loss_tr = all_losses.data.numpy()
    batch_losses = all_losses.clone()
    
    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]
    bmm_model_maxLoss = torch.FloatTensor([max_perc])
    bmm_model_minLoss = torch.FloatTensor([min_perc]) + 10e-6
    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)
    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4
    
    #FIT BMM on loss_tr
    bmm_model = BetaMixture1D(max_iters=30)
    bmm_model.fit(loss_tr)
    bmm_model.create_lookup(1)
    
    if epoch//10==0:
        epoch = "0"+str(epoch)
    else:
        epoch = str(epoch)

    plt.title('Epoch %s'%epoch)

    a = args.dataset.split('/')
    if len(a[-1])==0:
        a[-1] = a[-2]
    folder = os.path.join('figures',a[-1])
    if not os.path.isdir(folder):
        os.mkdir(folder)

    fname = os.path.join(folder,'bmm_%s.png'%epoch)
    
    bmm_model.plot(fname)
    
    #Get probabilities from BMM
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return bmm_model, bmm_model_maxLoss, bmm_model_minLoss, torch.FloatTensor(B).cuda()

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, save_path=None):
        x = np.linspace(0, 1, 100)
        plt.figure()
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()
        if save_path:
            plt.savefig(save_path)

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        #self.fc3 = torch.nn.Linear(self.input_size, self.output_size)
        

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc3(relu)
        #output = self.fc3(x)
        return output

class Model(nn.Module):
    def __init__(self, embedding, hidden_size=600, depth=6, dropout=0.3, cnn=False, nclasses=2):
        super(Model, self).__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs = dataloader.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id

        if cnn:
            self.encoder = modules.CNN_Text(
                self.emb_layer.n_d,
                widths = [3,4,5],
                filters=hidden_size
            )
            d_out = 3*hidden_size
        else:
            self.encoder = nn.LSTM(
                self.emb_layer.n_d,
                hidden_size//2,
                depth,
                dropout = dropout,
                # batch_first=True,
                bidirectional=True
            )
            d_out = hidden_size
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self, text, batch_size=32):
        batches_x = dataloader.create_batches_x(
            text,
            batch_size,
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                if self.cnn:
                    x = x.t()
                emb = self.emb_layer(x)

                if self.cnn:
                    output = self.encoder(emb)
                else:
                    output, hidden = self.encoder(emb)
                    output = torch.max(output, dim=0)[0]

                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)


def eval_model(niter, model, noise_model, input_x, input_y):
    model.eval()
    noise_model.eval()
    correct = 0.0
    cnt = 0.
    with torch.no_grad():
        for x, y in zip(input_x, input_y):
            x, y = Variable(x), Variable(y)
            #output = noise_model(model(x))
            output = model(x)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum()
            cnt += y.numel() 
    model.train()
    noise_model.train()
    return correct.item()/cnt



def train_model(epoch, model, noise_model, optimizer,
        train_x, train_y,
        test_x, test_y,
        best_test, save_path, bmm_model, nclasses):
    
    bmm_model, bmm_model_maxLoss, bmm_model_minLoss, prob = track_training_loss(model, noise_model, train_x, train_y, bmm_model, epoch)
    
    model.train()
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(reduce=False)
    kl_criterion = nn.KLDivLoss(reduction="none")
    softmax_criterion= nn.Softmax()
    log_softmax_criterion= nn.LogSoftmax()
    
    def contrastive_loss(clean_output, noisy_output, prob, y,epoch):
        #kl_loss = torch.sum(kl_criterion(log_softmax_criterion(noisy_output), softmax_criterion(clean_output)),axis=1)
        #contrastive_loss = torch.sum((1-prob)*kl_loss - prob*torch.clamp(kl_loss, min=0, max=1))
        if epoch < 5:
            return torch.zeros(1).cuda()
        batch_size=clean_output.size()[0]
        uniform_output=(1/(nclasses-1))*torch.ones(clean_output.size()).cuda()
        a= torch.linspace(0,batch_size-1,steps=batch_size).long()
        uniform_output[a,y]=0
        kl_loss = torch.sum(kl_criterion(log_softmax_criterion(clean_output),uniform_output),axis=1)
        contrastive_loss=torch.sum((1-prob)*criterion2(clean_output, y) + prob*kl_loss)
        #print(contrastive_loss)
        return contrastive_loss
    
    element_criterion = nn.CrossEntropyLoss(reduce=False)
    element_loss = []
    cnt=count_var=0
    beta=0
    total_contrast_loss=total_cross_entropy_loss=total_loss=0
    for x, y in zip(train_x, train_y):
        p= prob[count_var:count_var+x.size()[1]]
        count_var+=x.size()[1]
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y, p = Variable(x), Variable(y), Variable(p)
        clean_output = model(x)
        #noisy_output = noise_model(clean_output)
        noisy_output=clean_output
        contrast_loss = contrastive_loss(clean_output, noisy_output, p, y, epoch)
        cross_entropy_loss = criterion(noisy_output, y)
        loss = cross_entropy_loss + beta*contrast_loss
        total_contrast_loss+=contrast_loss.item()
        total_cross_entropy_loss+=cross_entropy_loss.item()
        total_loss+=loss.item()
        element_loss.extend(element_criterion(clean_output,y).cpu().detach().numpy().tolist())
        loss.backward()
        optimizer.step()
    test_acc = eval_model(niter, model, noise_model, test_x, test_y)

    sys.stdout.write("Epoch={} train_loss={:.6f} contrast_loss={:.6f} CE_loss={:.6f} dev_acc={:.6f}\n".format(
        epoch,
        total_loss, total_contrast_loss, total_cross_entropy_loss,
        test_acc
    ))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), os.path.join(save_path,"model.bin"))
    sys.stdout.write("\n")
    return best_test, element_loss

def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type+'.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))

def main(args):
    train_x, train_y, train_noise = dataloader.read_corpus(os.path.join(args.dataset,"train"), shuffle=True, get_noise=True)
    dev_x, dev_y, dev_noise = dataloader.read_corpus(os.path.join(args.dataset,"dev"), shuffle=True, get_noise=True)
    test_x, test_y = dataloader.read_corpus(os.path.join(args.dataset,"test"), shuffle=True)

    nclasses = max(train_y) + 1
    print("NUM CLASSES: "+ str(nclasses))
    model = Model(args.embedding, args.d, args.depth, args.dropout, args.cnn, nclasses=nclasses).cuda()
    noise_model = Feedforward(nclasses,10,nclasses).cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, list(model.parameters()) + list(noise_model.parameters())),
        lr = args.lr
    )

    train_x, train_y = dataloader.create_batches(
        train_x, train_y,
        args.batch_size,
        model.word2id,
    )
    
    dev_x, dev_y = dataloader.create_batches(
        dev_x, dev_y,
        args.batch_size,
        model.word2id,
    )
    
    test_x, test_y = dataloader.create_batches(
        test_x, test_y,
        args.batch_size,
        model.word2id,
    )

    best_dev = 0
    curr_best_dev=0
    early_stopping=0
    bmm_model = BetaMixture1D(max_iters=10)
    
    frames = []
    for epoch in range(args.max_epoch):
        best_dev, element_loss = train_model(epoch, model, noise_model, optimizer,
            train_x, train_y,
            dev_x, dev_y,
            best_dev, args.save_path,
            bmm_model, nclasses
        )
#         if curr_best_dev < best_dev:
#             curr_best_dev=best_dev
#             early_stopping=0
#             best_model=copy.deepcopy(model)
#         else:
#             early_stopping+=1
        
        #if early_stopping == 15:
        #    break
            
        if args.lr_decay>0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        frames.append(plot_histogram(element_loss, train_noise, epoch))
        create_gif(frames)
    
    test_acc = eval_model(args.max_epoch, model, noise_model, test_x, test_y)
    sys.stdout.write("test_err: {:.6f}\n".format(
        test_acc
    ))

    

def create_gif(frames):
    frames = [Image.open(i) for i in frames]
    a = args.dataset.split('/')
    if len(a[-1])==0:
        a[-1] = a[-2]
    folder = os.path.join('figures',a[-1])
    if not os.path.isdir(folder):
        os.mkdir(folder)

    fname = os.path.join(folder,'hist.gif')
    frames[0].save(fname, format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=150, loop=0)


def plot_histogram(losses, labels, epoch):
    clean = [x for x,y in zip(losses,labels) if y==0]
    noisy = [x for x,y in zip(losses,labels) if y==1]

    plt.figure()
    hist, bins = np.histogram(clean, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)

    hist, bins = np.histogram(noisy, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='red')

    if epoch//10==0:
        epoch = "0"+str(epoch)
    else:
        epoch = str(epoch)

    plt.title('Epoch %s'%epoch)

    a = args.dataset.split('/')
    if len(a[-1])==0:
        a[-1] = a[-2]
    folder = os.path.join('figures',a[-1])
    if not os.path.isdir(folder):
        os.mkdir(folder)

    fname = os.path.join(folder,'hist_%s.png'%epoch)

    plt.show()
    plt.savefig(fname)
    plt.clf()

    return fname


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--dataset", type=str, default="yelp", help="which dataset")
    argparser.add_argument("--embedding", type=str, default="counter-fitted-vectors.txt", help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=60)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='test')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print (args)
    torch.cuda.set_device(args.gpu_id)
    main(args)