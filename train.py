"""Training script"""

import os
import time
import copy
import shutil
import random

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model import SGRAF
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
    plot_gmm,
    save_loss_for_split,
)


def main(opt):

    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    best_rsum = 0
    start_epoch = 0
    best_rsum_warmup, best1, best2, best3, best4, best5, best6 = validate(opt, val_loader, [model_A, model_B])
    best_warmup_epoch = 0

    # save the history of losses from two networks
    all_loss = [[], []]

    # Warmup
    print("\n* Warmup")
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
            print("\nValidattion ...")
            validate(opt, val_loader, [model_A, model_B])
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        epoch = 0
        for epoch in range(0, opt.warmup_epoch):
            print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_A, epoch)
            print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_B, epoch)

            print("\nValidattion ...")
            rsum, b1, b2, b3, b4, b5, b6 = validate(opt, val_loader, [model_A, model_B])

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum_warmup and b1 > best1 and b2 > best2 and b3 > best3 and b4 > best4 and b5 > best5 and b6 > best6
            best_rsum_warmup = max(rsum, best_rsum_warmup)
            best1 = max(b1, best1)
            best2 = max(b2, best2)
            best3 = max(b3, best3)
            best4 = max(b4, best4)
            best5 = max(b5, best5)
            best6 = max(b6, best6)
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_A": model_A.state_dict(),
                        "model_B": model_B.state_dict(),
                        "best_rsum": best_rsum_warmup,
                        "opt": opt,
                    },
                    is_best = False, # this model is a warm up model
                    filename="warmup_model_{}.pth.tar".format(epoch),
                    prefix=opt.output_dir + "/",
                )
                best_warmup_epoch = epoch
            print("best warm up model is {}".format(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch)))

        # save the last warmup model
        save_checkpoint(
            {
                "epoch": epoch,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "opt": opt,
            },
            is_best = False, # this model is a warm up model
            filename="warmup_model_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )

        if os.path.isfile(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch)):
                checkpoint = torch.load(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch))
                model_A.load_state_dict(checkpoint["model_A"])
                model_B.load_state_dict(checkpoint["model_B"])
                print(
                    "=> load warmup checkpoint '{}' (epoch {})".format(
                        opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch), checkpoint["epoch"]
                    )
                )
        else:
            print(
                "=> no checkpoint found at '{}, no warming up is confirmed'".format(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch))
            )
            model_A = SGRAF(opt)
            model_B = SGRAF(opt)
        
        # evaluate on validation set
        print("\nValidattion ...")
        validate(opt, val_loader, [model_A, model_B])

    # save the history of losses from two networks
    all_loss = [[], []]
    print("\n* Co-training")

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (clean, hard, noisy)
        print("Split dataset ...")
        if epoch == start_epoch and os.path.isfile(opt.loss_for_split):
            loss_file = torch.load(opt.loss_for_split)
            prob_A = loss_file["prob_A"]
            print("loading probability of network A, shape is: {}".format(prob_A.shape))
            prob_B = loss_file["prob_B"]
            print("loading probability of network B, shape is: {}".format(prob_B.shape))
            all_loss = loss_file["all_loss"]
            print("loading all loss, len is: {}".format(len(all_loss)))
        
        else:    
            prob_A, prob_B, all_loss = eval_train(
                opt,
                model_A,
                model_B,
                noisy_trainloader,
                data_size,
                all_loss,
                clean_labels,
                epoch,
            )

        clean, hard, noisy = split_samples(prob_A, prob_B, opt.p_threshold)

        # clean = np.ones(clean.shape, dtype=bool)
        # hard = np.ones(hard.shape, dtype=bool)
        # noisy = np.ones(noisy.shape, dtype=bool)

        clean_ratio_c = np.sum(clean_labels[clean.nonzero()[0]]) / np.sum(clean)
        clean_ratio_h = np.sum(clean_labels[hard.nonzero()[0]]) / np.sum(hard) 
        clean_ratio_n = np.sum(clean_labels[noisy.nonzero()[0]]) / np.sum(noisy)      

        print("clean split has {} pairs and {} pairs are GT clean. The clean ratio is {}".format(np.sum(clean), np.sum(clean_labels[clean.nonzero()[0]]), clean_ratio_c))
        print("hard split has {} pairs and {} pairs are GT clean. The clean ratio is {}".format(np.sum(hard), np.sum(clean_labels[hard.nonzero()[0]]), clean_ratio_h))
        print("noisy split has {} pairs and {} pairs are GT clean. The clean ratio is {}".format(np.sum(noisy), np.sum(clean_labels[noisy.nonzero()[0]]), clean_ratio_n))
        print("we totally have {} GT clean pairs and the number of training pairs is {}. \
            the real noise_ratio is {}, the setting is {}".format(np.sum(clean_labels), len(captions_train), 1 - (np.sum(clean_labels) / len(captions_train)), opt.noise_ratio))

        ratio = np.sum(clean) / (np.sum(clean)+np.sum(hard)+np.sum(noisy))

        print("\nModel A training ...")
        # train model_A
        clean_data_trainloader, hard_data_trainloader, noisy_data_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            clean=clean,
            hard=hard,
            noisy=noisy,
            prob_A=prob_A,
            prob_B=prob_B,
        )
        train_one_epoch(opt, model_A, model_B, clean_data_trainloader, hard_data_trainloader, noisy_data_trainloader, epoch, model="A", ratio=ratio)

        print("\nModel B training ...")
        # train model_B
        clean_data_trainloader, hard_data_trainloader, noisy_data_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            clean=clean,
            hard=hard,
            noisy=noisy,
            prob_A=prob_A,
            prob_B=prob_B,
        )
        train_one_epoch(opt, model_B, model_A, clean_data_trainloader, hard_data_trainloader, noisy_data_trainloader, epoch, model="B", ratio=ratio)

        print("\nValidattion ...")
        # evaluate on validation set
        rsum, bb1, bb2, bb3, bb4, bb5, bb6= validate(opt, val_loader, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # save the first stage's checkpoint
        elif epoch == opt.hard_start_epoch - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )
        
        # save the second stage's checkpoint
        elif epoch >= opt.noisy_start_epoch - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # save the last checkpoint
        elif epoch == opt.num_epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )


def train_one_epoch(opt, net, net2, clean_loader, hard_loader=None, noisy_loader=None, epoch=None, model="", ratio=None):
    """
    One epoch training.
    """

    if len(clean_loader) == 0:
        print("No clean pairs! This {} epoch is skipped!".format(epoch))
        return

    clean_labels = AverageMeter("clean labels")
    hard_labels = AverageMeter("hard labels")
    noisy_labels = AverageMeter("noisy labels")
    filters = AverageMeter("filter:")
    row_mean_c = AverageMeter("clean row mean:")
    col_mean_c = AverageMeter("clean col mean:")
    row_mean_h = AverageMeter("hard row mean:")
    col_mean_h = AverageMeter("hard col mean:")
    row_mean_n = AverageMeter("noisy row mean:")
    col_mean_n = AverageMeter("noisy col mean:")
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(clean_loader),
        [batch_time, data_time, losses, clean_labels, hard_labels, noisy_labels, filters, row_mean_c, col_mean_c, row_mean_h, col_mean_h, row_mean_n, col_mean_n],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    if hard_loader is not None and len(hard_loader) != 0:
        hard_iter = iter(hard_loader)
    
    if noisy_loader is not None and len(noisy_loader) != 0:
        noisy_iter = iter(noisy_loader)

    labels_c = []
    # pred_labels_c = []

    labels_h = []
    # pred_labels_h = []

    labels_n = []
    # pred_labels_n = []

    end = time.time()
    for i, batch_train_data in enumerate(clean_loader):
        (
            batch_images_c,
            batch_text_c,
            batch_lengths_c,
            c_ids,
            batch_labels_c,
            batch_prob_A_c,
            batch_prob_B_c,
            _c_correspondence
        ) = batch_train_data
        
        labels_c.append(batch_labels_c)

        skip_hard_train = False # loader drop last
        skip_noisy_train = False
        
        # hard data
        if hard_loader is not None and len(hard_loader) != 0:
            try:
                (
                    batch_images_h,
                    batch_text_h,
                    batch_lengths_h,
                    h_ids,
                    batch_labels_h,
                    batch_prob_A_h,
                    batch_prob_B_h,
                    _h_correspondence
                ) = hard_iter.next()
            except:
                skip_hard_train = True
            labels_h.append(batch_labels_h)

        # noisy data
        if noisy_loader is not None and len(noisy_loader) != 0:
            try:
                (
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    n_ids,
                    batch_labels_n,
                    _n_correspondence
                ) = noisy_iter.next()
            except:
                noisy_iter = iter(noisy_loader)
                ( # whether skip or not
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    n_ids,
                    batch_labels_n,
                    _n_correspondence
                ) = noisy_iter.next()
            labels_n.append(batch_labels_n)

        # measure data loading time
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if batch_images_c.size(0) == 1:
            break
        else:
            if batch_images_h.size(0) == 1:
                skip_hard_train = True
            if batch_images_n.size(0) == 1:
                skip_noisy_train = True

        # loading cuda
        if torch.cuda.is_available():
            batch_prob_A_c = batch_prob_A_c.cuda()
            batch_prob_B_c = batch_prob_B_c.cuda()
            batch_labels_c = batch_labels_c.cuda()

            batch_prob_A_h = batch_prob_A_h.cuda()
            batch_prob_B_h = batch_prob_B_h.cuda()
            batch_labels_h = batch_labels_h.cuda()

        # label refinement
        with torch.no_grad():
            net.val_start()
            # clean data
            if model == "A":
                predict_clean = net.predict(batch_images_c, batch_text_c, batch_lengths_c)
                targets_c = torch.mul(batch_prob_B_c, batch_labels_c) + torch.mul((1 - batch_prob_B_c), predict_clean.t())
            
            elif model == "B":
                predict_clean = net.predict(batch_images_c, batch_text_c, batch_lengths_c)
                targets_c = torch.mul(batch_prob_A_c, batch_labels_c) + torch.mul((1 - batch_prob_A_c), predict_clean.t())

            clean_labels.update(np.mean(targets_c.cpu().numpy()), batch_images_c.size(0))

            # hard data
            predict_hard = net.predict(batch_images_h, batch_text_h, batch_lengths_h)
            targets_h = torch.mul((batch_prob_A_h + batch_prob_B_h) / 2, batch_labels_h) + torch.mul((1 - (batch_prob_A_h + batch_prob_B_h) / 2), predict_hard.t())
            hard_labels.update(np.mean(targets_h.cpu().numpy()), batch_images_h.size(0))

            # noisy data
            pu1 = net.predict(batch_images_n, batch_text_n, batch_lengths_n)
            pu2 = net2.predict(batch_images_n, batch_text_n, batch_lengths_n)
            targets_u = (pu1.t() + pu2.t()) / 2
            noisy_labels.update(np.mean(targets_u.cpu().numpy()), batch_images_n.size(0))

            filter = (np.mean(targets_c.cpu().numpy()) * len(clean_loader.dataset) + np.mean(targets_h.cpu().numpy()) * \
                len(hard_loader.dataset) + np.mean(targets_u.cpu().numpy()) * len(noisy_loader.dataset)) / (len(clean_loader.dataset) + len(hard_loader.dataset) + len(noisy_loader.dataset))

            filters.update(filter, batch_images_c.size(0) + batch_images_n.size(0) + batch_images_h.size(0))

            if ratio > 0.85:
                filter_c = 0
                filter_h = 0
            else:
                filter_c = max(opt.hard_start_epoch-epoch, filter)
                filter_h = filter

        net.train_start()
        # train with clean + hard + noisy data 
        loss_c, r_mean_c, c_mean_c = net.train(
            batch_images_c,
            batch_text_c,
            batch_lengths_c,
            labels=targets_c,
            mode="train",
            soft_label=opt.soft_label,
            smooth_label=opt.smooth_label,
            filter=filter_c
        )
        # loss_c = 0

        row_mean_c.update(r_mean_c.item(), batch_images_c.size(0))
        col_mean_c.update(c_mean_c.item(), batch_images_c.size(0))

        # Training with difficult samples too early is harmful to the model.
        
        if epoch < opt.hard_start_epoch:
            loss_h = 0
            loss_n = 0

        elif epoch < opt.noisy_start_epoch:
            loss_h = 0
            loss_n = 0
            if skip_hard_train is not True:
                loss_h, r_mean_h, c_mean_h = net.train(
                    batch_images_h, 
                    batch_text_h,
                    batch_lengths_h,
                    labels=targets_h,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter_h
                )
                row_mean_h.update(r_mean_h.item(), batch_images_h.size(0))
                col_mean_h.update(c_mean_h.item(), batch_images_h.size(0))
        else:
            loss_h = 0
            loss_n = 0

            if skip_hard_train is not True:

                loss_h, r_mean_h, c_mean_h = net.train(
                    batch_images_h, 
                    batch_text_h,
                    batch_lengths_h,
                    labels=targets_h,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter_h
                )
                row_mean_h.update(r_mean_h.item(), batch_images_h.size(0))
                col_mean_h.update(c_mean_h.item(), batch_images_h.size(0))

            if skip_noisy_train is not True:

                loss_n, r_mean_n, c_mean_n = net.train(
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    labels=targets_u,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter
                )
                row_mean_n.update(r_mean_n.item(), batch_images_n.size(0))
                col_mean_n.update(c_mean_n.item(), batch_images_n.size(0))

        loss = loss_c + loss_h + loss_n
        losses.update(loss, batch_images_c.size(0) + batch_images_n.size(0) + batch_images_h.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss, _, _ = model.train(images, captions, lengths, mode="warmup")
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum, r1, r5, r10, r1i, r5i, r10i


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and probability
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    captions_A = [[] for i in range(data_size)]
    captions_B = [[] for i in range(data_size)]


    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]
                captions_A[ids[b]].append(captions[b].cpu().numpy())
                captions_B[ids[b]].append(captions[b].cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    # NaN, because of dividing by 0 (losses_A.max() - losses_A.min()) is a very small value
    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    print("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()] # The probability of belonging to the first group of components (have smaller loss).

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    print("The length of computed loss of network A is: {}".format(len(input_loss_A.cpu().numpy())))
    print("The length of computed loss of network B is: {}".format(len(input_loss_B.cpu().numpy())))

    draw = opt.draw_gmm
    if draw:
        plot_gmm(epoch, gmm_A, input_loss_A.cpu().numpy(), clean_labels, opt.output_dir + "/" + opt.data_name + '_noise_ratio_' + str(opt.noise_ratio) + '_epoch_' + str(epoch) + '_gmm_A.png')
        plot_gmm(epoch, gmm_B, input_loss_B.cpu().numpy(), clean_labels, opt.output_dir + "/" + opt.data_name + '_noise_ratio_' + str(opt.noise_ratio) + '_epoch_' + str(epoch) + '_gmm_B.png')

    # Save the calculated loss for split dataset.
    save_loss = True
    if save_loss:
        save_loss_for_split(
                {
                    "epoch": epoch,
                    "prob_A": prob_A,
                    "prob_B": prob_B,
                    "all_loss": all_loss,
                    "opt": opt,
                },
                filename="split_loss_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred


def split_samples(prob_A, prob_B, threshold):

    clean_mask_A = split_prob(prob_A, threshold)
    clean_mask_B = split_prob(prob_B, threshold)

    # after split clean and noisy using split_prob(), we have two lists of splits 
    clean_mask = np.zeros(prob_A.shape[0], dtype=bool) 
    clean_mask[np.logical_and(clean_mask_A == True, clean_mask_B == True)] = True

    noisy_mask = np.zeros(prob_A.shape[0], dtype=bool)
    noisy_mask[np.logical_and(clean_mask_A == False, clean_mask_B == False)] = True

    hard_mask = np.zeros(prob_A.shape[0], dtype=bool)
    hard_mask[np.logical_or(np.logical_and(clean_mask_A == True, clean_mask_B == False),
     np.logical_and(clean_mask_A == False, clean_mask_B == True))] = True
    
    return clean_mask, hard_mask, noisy_mask
