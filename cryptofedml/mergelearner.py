from __future__ import division, print_function, absolute_import

import os
import pdb
import copy
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from learner import Learner
from metalearner import MetaLearner
from dataloader import prepare_data
from utils import *

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode', choices=['train', 'test'], default='train')
# Hyper-parameters
FLAGS.add_argument('--n-shot', type=int, default=5,
                   help="How many examples per class for training (k, n_support)")
FLAGS.add_argument('--n-eval', type=int, default=15,
                   help="How many examples per class for evaluation (n_query)")
FLAGS.add_argument('--n-class', type=int, default=5,
                   help="How many classes (N, n_way)")
FLAGS.add_argument('--input-size', type=int, default=4,
                   help="Input size for the first LSTM")
FLAGS.add_argument('--hidden-size', type=int, default=4,
                   help="Hidden size for the first LSTM")
FLAGS.add_argument('--lr', type=float, default=1e-3,
                   help="Learning rate")
FLAGS.add_argument('--episode', type=int, default=50000,
                   help="Episodes to train")
FLAGS.add_argument('--episode-val', type=int, default=100,
                   help="Episodes to eval")
FLAGS.add_argument('--epoch', type=int, default=8,
                   help="Epoch to train for an episode")
FLAGS.add_argument('--batch-size', type=int, default=25,
                   help="Batch size when training an episode")
FLAGS.add_argument('--image-size', type=int, default=84,
                   help="Resize image to this size")
FLAGS.add_argument('--grad-clip', type=float, default=0.25,
                   help="Clip gradients larger than this number")
FLAGS.add_argument('--bn-momentum', type=float, default=0.95,
                   help="Momentum parameter in BatchNorm2d")
FLAGS.add_argument('--bn-eps', type=float, default=1e-3,
                   help="Eps parameter in BatchNorm2d")

# Paths
FLAGS.add_argument('--data', choices=['miniimagenet'], default='miniimagenet',
                   help="Name of dataset")
FLAGS.add_argument('--data-root', type=str, default='bob/',
                   help="Location of data")
FLAGS.add_argument('--resume', type=str,
                   help="Location to pth.tar")
FLAGS.add_argument('--save', type=str, default='logs',
                   help="Location to logs and ckpts")
# Others
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', type=bool, default=True,
                   help="DataLoader pin_memory")
FLAGS.add_argument('--log-freq', type=int, default=50,
                   help="Logging frequency")
FLAGS.add_argument('--val-freq', type=int, default=1000,
                   help="Validation frequency")
FLAGS.add_argument('--seed', type=int,
                   help="Random seed")

from dp_sgd import DP_SGD
from dp_Adam import DP_Adam

def meta_test(eps, eval_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger):
    for subeps, (episode_x, episode_y) in enumerate(tqdm(eval_loader, ascii=True)):
        train_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(
            args.dev)  # [n_class * n_shot, :]
        train_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev)  # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev)  # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev)  # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.eval()
        cI = train_cslearner(learner_w_grad, metalearner, train_input, train_target, args)

        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)

        logger.batch_info(loss=loss.item(), acc=acc, phase='eval')

    return logger.batch_info(eps=eps, totaleps=args.episode_val, phase='evaldone')


def train_cslearner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    for _ in range(args.epoch):
        for i in range(0, len(train_input), args.batch_size):
            x = train_input[i:i + args.batch_size]
            y = train_target[i:i + args.batch_size]

            # get the loss/grad
            learner_w_grad.copy_flat_params(cI)
            output = learner_w_grad(x)
            loss = learner_w_grad.criterion(output, y)
            acc = accuracy(output, y)
            learner_w_grad.zero_grad()
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in learner_w_grad.parameters()], 0)

            # preprocess grad & loss and metalearner forward
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0))  # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            # print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))

    return cI


def train_learner(alicelner, boblner, metalearner, atrain_input, atrain_target, btrain_input, btrain_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    aliceloss = 0
    aliceacc = 0
    bobloss = 0
    bobacc = 0
    #abdatafile = open('sgdab.txt', 'w')
    for _ in range(args.epoch):
        for i in range(0, len(atrain_input), args.batch_size):
            ax = atrain_input[i:i + args.batch_size]
            ay = atrain_target[i:i + args.batch_size]

            bx = btrain_input[i:i + args.batch_size]
            by = btrain_target[i:i + args.batch_size]

            # get the loss/grad
            alicelner.copy_flat_params(cI)
            aoutput = alicelner(ax)
            aloss = alicelner.criterion(aoutput, ay)
            a_acc = accuracy(aoutput, ay)
            alicelner.zero_grad()
            aloss.backward()
            agrad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in alicelner.parameters()], 0)
            aliceloss = aloss
            aliceacc = a_acc

            # get the loss/grad
            boblner.copy_flat_params(cI)
            boutput = boblner(bx)
            bloss = boblner.criterion(boutput, by)
            b_acc = accuracy(boutput, by)
            boblner.zero_grad()
            bloss.backward()
            bgrad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in boblner.parameters()], 0)
            bobloss = bloss
            bobacc = b_acc

            # preprocess grad & loss and metalearner forward
            grad = torch.add(agrad, bgrad)/2
            loss = torch.add(aloss, bloss)/2
            grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0))  # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

            # print("training loss: {:8.6f} acc: {:6.3f}, mean grad: {:8.6f}".format(loss, acc, torch.mean(grad)))
    #print("%f %f %f %f", aliceloss, aliceacc, bobloss, bobacc)
    return cI, aliceloss, aliceacc, bobloss, bobacc


def main():
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    if args.seed is None:
        args.seed = random.randint(0, 1e3)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        args.dev = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.dev = torch.device('cuda')

    logger = GOATLogger(args)

    # Get data
    train_loader, train1_loader, val_loader, val1_loader, test_loader = prepare_data(args)

    # Set up learner, meta-learner
    learner_w_grad = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    alicelner = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    boblner = Learner(args.image_size, args.bn_eps, args.bn_momentum, args.n_class).to(args.dev)
    alicelner.load_state_dict(torch.load('./adamalicelearner.pth'))
    boblner.load_state_dict(torch.load('./adamboblearner.pth'))

    for param in alicelner.state_dict():
        learner_w_grad.state_dict()[param] = \
            torch.add(alicelner.state_dict()[param], boblner.state_dict()[param])/2

    learner_wo_grad = copy.deepcopy(learner_w_grad)
    metalearner = MetaLearner(args.input_size, args.hidden_size, learner_w_grad.get_flat_params().size(0)).to(args.dev)
    alicemetalner = MetaLearner(args.input_size, args.hidden_size, alicelner.get_flat_params().size(0)).to(args.dev)
    alicemetalner.load_state_dict(torch.load('./adamalicemetalearner.pth'))
    bobmetalner = MetaLearner(args.input_size, args.hidden_size, boblner.get_flat_params().size(0)).to(args.dev)
    bobmetalner.load_state_dict(torch.load('./adambobmetalearner.pth'))

    for param in alicemetalner.state_dict():
        metalearner.state_dict()[param] = \
            torch.add(alicemetalner.state_dict()[param], bobmetalner.state_dict()[param])/2

    # metalearner.load_state_dict(torch.load())
    metalearner.metalstm.init_cI(learner_w_grad.get_flat_params())

    # Set up loss, optimizer, learning rate scheduler
    #optim = torch.optim.Adam(metalearner.parameters(), args.lr)
    #optim = torch.optim.SGD(metalearner.parameters(), args.lr)
    #optim = DP_SGD(metalearner.parameters(), args.lr)
    optim = DP_Adam(metalearner.parameters(), args.lr)
    if args.resume:
        logger.loginfo("Initialized from: {}".format(args.resume))
        last_eps, metalearner, optim = resume_ckpt(metalearner, optim, args.resume, args.dev)

    if args.mode == 'test':
        _ = meta_test(last_eps, test_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
        return

    best_acc = 0.0

    logger.loginfo("Start training")
    # Meta-training
    logfile = open('adamnew.txt', 'w')

    print("a")
    print("a")
    for eps, ((episode_x, episode_y), (bobepisode_x, bobepisode_y)) in enumerate(zip(train_loader, train1_loader)):
        # episode_x.shape = [n_class, n_shot + n_eval, c, h, w]
        # episode_y.shape = [n_class, n_shot + n_eval] --> NEVER USED
        print("fds")
        print("fa")
        atrain_input = episode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(
            args.dev)  # [n_class * n_shot, :]
        atrain_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev)  # [n_class * n_shot]
        btrain_input = bobepisode_x[:, :args.n_shot].reshape(-1, *episode_x.shape[-3:]).to(
            args.dev)  # [n_class * n_shot, :]
        btrain_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_shot)).to(args.dev)  # [n_class * n_shot]
        test_input = episode_x[:, args.n_shot:].reshape(-1, *episode_x.shape[-3:]).to(args.dev)  # [n_class * n_eval, :]
        test_target = torch.LongTensor(np.repeat(range(args.n_class), args.n_eval)).to(args.dev)  # [n_class * n_eval]

        # Train learner with metalearner
        learner_w_grad.reset_batch_stats()
        learner_wo_grad.reset_batch_stats()
        learner_w_grad.train()
        learner_wo_grad.train()
        cI, aloss, aacc, bloss, bacc = train_learner(alicelner, boblner, metalearner, atrain_input,
                           atrain_target, btrain_input, btrain_target, args)
        if eps % 500 == 0:
            logfile.write(str(aloss) + " " + str(aacc) + " " + str(bloss) + " " + str(bacc))
            logfile.write("\n")
        # Train meta-learner with validation loss
        learner_wo_grad.transfer_params(learner_w_grad, cI)
        output = learner_wo_grad(test_input)
        loss = learner_wo_grad.criterion(output, test_target)
        acc = accuracy(output, test_target)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(metalearner.parameters(), args.grad_clip)
        #optim.step()
        #optim.dpsgd_step(eps)
        optim.dpadam_step(eps)

        logger.batch_info(eps=eps, totaleps=args.episode, loss=loss.item(), acc=acc, phase='train')

        # Meta-validation
        if eps % args.val_freq == 0 and eps != 0:
            save_ckpt(eps, metalearner, optim, args.save)
            acc = meta_test(eps, val_loader, learner_w_grad, learner_wo_grad, metalearner, args, logger)
            if acc > best_acc:
                best_acc = acc
                logger.loginfo("* Best accuracy so far *\n")

    logger.loginfo("Done")

    mergelearner_dict = learner_w_grad.state_dict()
    torch.save(mergelearner_dict, './newmergelearner.pth')
    mergemetalearner_dict = metalearner.state_dict()
    torch.save(mergemetalearner_dict, './newmergemetalearner.pth')
    logfile.close()


if __name__ == '__main__':
    main()
