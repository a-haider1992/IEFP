import sys
import argparse
import random
import warnings
import os
from itertools import chain
import math
import copy
import time
import numpy as np
import itertools

from fr_ae_mi_models import fr_ae_mi_model_1
from fr_models import fr_model_1
from custom_datasets import ImageFolderWithAgeGroup
from custom_datasets_2 import Dataset_floder
from meta import age_cutoffs, ms1mv3, FGNET, processed_CACD_VS, processed_CALFW
from meta import MORPH2_probe_3000, MORPH2_gallery_3000, MORPH2_probe_10000, MORPH2_gallery_10000
from utils import multi_accuracies, accuracy_percent, calculate_roc_auc_eer, argmax_mae
from pre_process import image_test_1crop_transform, image_test_2crops_transform
from datasetV3 import TrainingData
# from utils_amp import MaxClipGradScaler


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
import pdb


parser = argparse.ArgumentParser(
    description='PyTorch age-invarinat face recognition')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=42, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', default=[20, 30, 40, 50], nargs='+', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--output_dir', type=str, default='ms1mv3_iresnet50_frspu_ae_mi',
                    help="output directory of our model (in ../snapshot directory)")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--feature_dim', default=512, type=int)
# MTLVIFR args
parser.add_argument('--lfw_mode', default=0, type=int)
parser.add_argument('--eval_iter', default=100, type=int)

parser.add_argument('--spu_scale', default=1, type=int)
parser.add_argument('--fp16', default=False, type=bool)

# options for pure face recognition
parser.add_argument('--face-id-loss', default='cosface',
                    type=str, choices=['cosface', 'arcface'])
parser.add_argument('--scale', default=64.0, type=float, metavar='M',
                    help='scaler in arcface or cosface')
parser.add_argument('--margin', default=0.35, type=float, metavar='M',
                    help='angular margin in arcface or cosine margin in cosface')
parser.add_argument('--lambda1', default=0.001, type=float)
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    # pdb.set_trace()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def calculateMetrics(a, b):
    # Compute Euclidean distance
    euclidean_distance = torch.norm(a - b, p=1)

    # Compute cosine similarity
    cosine_similarity = F.cosine_similarity(a, b).item()

    # Compute correlation coefficient
    cov = torch.mean((a - torch.mean(a)) * (b - torch.mean(b)))
    std_a = torch.std(a)
    std_b = torch.std(b)
    corr_coef = cov / (std_a * std_b)
    corr_coef = corr_coef.item()

    # Compute mean squared error
    mean_squared_error = F.mse_loss(a, b)
    return euclidean_distance, cosine_similarity, corr_coef, mean_squared_error


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = fr_ae_mi_model_1(n_cls=93431, args=args)
    # model = fr_model_1(n_cls=93431, args=args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    pretrained_model_dict = './pretrained_IEFP_models/ms1mv3_iresnet50_IEFP.pth'
    model.load_state_dict(torch.load(pretrained_model_dict))

    # cudnn.benchmark = True

    # Data loading code

    # Loading LFW only

    print("Loading LFW dataset or CASIA-WebFACE..")
    if args.lfw_mode == 0:
        train_lfw_dataset = TrainingData(
            'pairs.csv', mode=0, transform=image_test_2crops_transform)
    elif args.lfw_mode == 1:
        train_lfw_dataset = TrainingData(
            'pairs_mismatch.csv', mode=1, transform=image_test_2crops_transform)
    weights = None
    loader = torch.utils.data.DataLoader(
        train_lfw_dataset, shuffle=False, batch_size=args.batch_size)
    loader = iter(loader)
    total_iter = int(args.eval_iter)
    total_correct_pred = 0
    total_incorrect_pred = 0
    mean_euc_dis = mean_cos_sim = mean_corr_coeff = mean_mse = 0.0
    # pdb.set_trace()
    with torch.no_grad():
        for _ in range(0, total_iter):
            xs_a, xs_b = next(loader)
            if args.gpu is not None and torch.cuda.is_available():
                xs_a = xs_a.cuda(args.gpu, non_blocking=True)
                xs_b = xs_b.cuda(args.gpu, non_blocking=True)
            # pdb.set_trace()
            xs_a_embedding, _, _, _ = model(xs_a)
            xs_b_embedding, _, _, _ = model(xs_b)
            xs_a_embedding = xs_a_embedding[:, :512]
            xs_b_embedding = xs_b_embedding[:, :512]
            euc_dis, cos_sim, corr_coeff, mse = calculateMetrics(
                xs_a_embedding, xs_b_embedding)
            mean_euc_dis += euc_dis
            mean_cos_sim += cos_sim
            mean_corr_coeff += corr_coeff
            mean_mse += mse
            # if similarity_error <= 1e-4 or mean_corr >= 0.5:
            #     total_correct_pred += 1
            # else:
            #     total_incorrect_pred += 1
            if args.lfw_mode == 0:
                if cos_sim == 1.0:
                    total_correct_pred += 1
                else:
                    total_incorrect_pred += 1
            elif args.lfw_mode == 1:
                if cos_sim <= 0.005:
                    total_correct_pred += 1
                else:
                    total_incorrect_pred += 1
        mean_euc_dis, mean_cos_sim, mean_corr_coeff, mean_mse = mean_euc_dis / total_iter, mean_cos_sim / total_iter, mean_corr_coeff / total_iter, mean_mse / total_iter
        print(f'The mean euclidean distance: {mean_euc_dis}')
        print(f'The mean cosine similarity: {mean_cos_sim}')
        print(f'The mean correlation: {mean_corr_coeff}')
        print(f'The mean MSE: {mean_mse}')
        if args.lfw_mode == 0:
            # True or same samples
            true_negatives = total_incorrect_pred
            true_positives = total_correct_pred
            print(
                f'The same LFW samples got {true_negatives} True Negatives, and {true_positives} True Positives')
        elif args.lfw_mode == 1:
            # False or different samples
            false_positives = total_incorrect_pred
            false_negatives = total_correct_pred
            print(
                f'The different LFW samples got {false_negatives} False Negatives, and {false_positives} False Positives')

    # test_2ds_FGNET = [ImageFolderWithAgeGroup(FGNET['pat'], FGNET['pos'], age_cutoffs, FGNET['file_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_CACD_VS = [Dataset_floder(processed_CACD_VS['file_root'], processed_CACD_VS['pair_lists_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_CALFW = [Dataset_floder(processed_CALFW['file_root'], processed_CALFW['pair_lists_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_morph2_probe_3000 = [ImageFolderWithAgeGroup(MORPH2_probe_3000['pat'],
    #             MORPH2_probe_3000['pos'], age_cutoffs, MORPH2_probe_3000['file_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_morph2_gallery_3000 = [ImageFolderWithAgeGroup(MORPH2_gallery_3000['pat'],
    #             MORPH2_gallery_3000['pos'], age_cutoffs, MORPH2_gallery_3000['file_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_morph2_probe_10000 = [ImageFolderWithAgeGroup(MORPH2_probe_10000['pat'],
    #             MORPH2_probe_10000['pos'], age_cutoffs, MORPH2_probe_10000['file_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]
    # test_2ds_morph2_gallery_10000 = [ImageFolderWithAgeGroup(MORPH2_gallery_10000['pat'],
    #             MORPH2_gallery_10000['pos'], age_cutoffs, MORPH2_gallery_10000['file_root'],
    #             transform=image_test_2crops_transform(resize_size=112, crop_size=112,
    #                             mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])[i]) for i in range(2)]

    # test_2ld_FGNET = [torch.utils.data.DataLoader(
    #     test_2ds_FGNET[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_CACD_VS = [torch.utils.data.DataLoader(
    #     test_2ds_CACD_VS[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_CALFW = [torch.utils.data.DataLoader(
    #     test_2ds_CALFW[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_morph2_probe_3000 = [torch.utils.data.DataLoader(
    #     test_2ds_morph2_probe_3000[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_morph2_gallery_3000 = [torch.utils.data.DataLoader(
    #     test_2ds_morph2_gallery_3000[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_morph2_probe_10000 = [torch.utils.data.DataLoader(
    #     test_2ds_morph2_probe_10000[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]
    # test_2ld_morph2_gallery_10000 = [torch.utils.data.DataLoader(
    #     test_2ds_morph2_gallery_10000[i], shuffle=False, batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True) for i in range(2)]

    # if not os.path.exists('snapshot/' + args.output_dir):
    #     os.makedirs('snapshot/' + args.output_dir, exist_ok=True)
    # log_file = 'snapshot/' + args.output_dir + '/log_test.csv'

    # save_path = 'saved_models/' + args.output_dir
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)

    # with open(log_file, 'a+') as f:
    #     f.write('\n')
    #     f.write('start recording\n')

    #     # evaluate on test set

    # test_fi_ae_FGNET(test_2ld_FGNET, model, log_file, args, crop_number=2, crops_mix_mode='concatenate')
    # test_fi_ae_FGNET(test_2ld_FGNET, model, log_file, args, crop_number=2, crops_mix_mode='add')
    # test_face_verification(test_2ld_CALFW, model, 'CALFW', log_file, args, crop_number=2, crops_mix_mode='concatenate')
    # test_face_verification(test_2ld_CALFW, model, 'CALFW', log_file, args, crop_number=2, crops_mix_mode='add')
    # test_face_identification_morph2(test_2ld_morph2_probe_3000, test_2ld_morph2_gallery_3000,
    #                                     test_2ld_morph2_probe_10000, test_2ld_morph2_gallery_10000,
    #                                     model, log_file, args, crop_number=2, crops_mix_mode='concatenate')
    # test_face_identification_morph2(test_2ld_morph2_probe_3000, test_2ld_morph2_gallery_3000,
    #                                     test_2ld_morph2_probe_10000, test_2ld_morph2_gallery_10000,
    #                                     model, log_file, args, crop_number=2, crops_mix_mode='add')

    # test_face_verification(test_2ld_CACD_VS, model, 'CACD_VS', log_file, args, crop_number=2, crops_mix_mode='concatenate')
    # test_face_verification(test_2ld_CACD_VS, model, 'CACD_VS', log_file, args, crop_number=2, crops_mix_mode='add')

    # with open(log_file,'a+') as f:
    #     f.write('log_over\n')

    dist.destroy_process_group()


def test_fi_ae_FGNET(test_loaders, model, log_file, args, crop_number=2, crops_mix_mode='concatenate'):
    mi_est = AverageMeter('mi_estimation', ':.2f')
    loglikeli = AverageMeter('loglikelihood of CLUB-net', '.2f')
    argmax_MAE = AverageMeter('argmax_MAE', ':.2f')
    model.train(False)
    with torch.no_grad():
        loader_list = [iter(test_loaders[i]) for i in range(crop_number)]
        for j in range(len(loader_list[0])):
            for k in range(crop_number):
                xs, ys, real_age_range = loader_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs = xs.cuda(args.gpu, non_blocking=True)
                bs_xs_embedding, bs_age_logits, bs_mi_estimation, bs_loglikelihood = model(
                    xs, emb=True)
                if k == 0:
                    mixed_bs_xs_embedding = bs_xs_embedding
                    mixed_bs_age_probs = F.softmax(bs_age_logits, dim=1)
                    mixed_bs_mi_estimation = bs_mi_estimation
                    mixed_bs_loglikelihood = bs_loglikelihood
                else:
                    if crops_mix_mode == 'concatenate':
                        mixed_bs_xs_embedding = torch.cat(
                            (mixed_bs_xs_embedding, bs_xs_embedding), dim=1)
                    else:
                        mixed_bs_xs_embedding = torch.add(
                            mixed_bs_xs_embedding, bs_xs_embedding)
                    mixed_bs_age_probs += F.softmax(bs_age_logits, dim=1)
                    mixed_bs_mi_estimation += bs_mi_estimation
                    mixed_bs_loglikelihood += bs_loglikelihood
            mixed_bs_age_probs = mixed_bs_age_probs / float(crop_number)
            mixed_bs_mi_estimation = mixed_bs_mi_estimation / \
                float(crop_number)
            mixed_bs_loglikelihood = mixed_bs_loglikelihood / \
                float(crop_number)
            real_batch_size = len(ys)
            bs_argmax_MAE = argmax_mae(
                mixed_bs_age_probs.cpu(), real_age_range)
            argmax_MAE.update(bs_argmax_MAE.item(), real_batch_size)
            mi_est.update(mixed_bs_mi_estimation.item(), real_batch_size)
            loglikeli.update(mixed_bs_loglikelihood.item(), real_batch_size)

            if j == 0:
                total_xs_embedding = mixed_bs_xs_embedding
                gt_labels = ys
            else:
                total_xs_embedding = torch.cat(
                    (total_xs_embedding, mixed_bs_xs_embedding), dim=0)
                gt_labels = torch.cat((gt_labels, ys), dim=0)

        total_xs_embedding = F.normalize(total_xs_embedding.cpu(), p=2, dim=1)

        ##############
        # calculate the face identification rank-1 accuracy on under LOPO protocol
        logits = total_xs_embedding.mm(total_xs_embedding.t())
        mask = 1.0 - torch.eye(logits.size(0))
        logits = logits.mul(mask)
        _, pred_pos = logits.topk(1, dim=1, largest=True, sorted=True)
        pred_pos = pred_pos.squeeze()
        pred_labels = torch.zeros_like(gt_labels)
        for i in range(len(gt_labels)):
            pred_labels[i] = gt_labels[pred_pos[i]]
        fr_acc_LOPO = accuracy_percent(pred_labels, gt_labels)
        ##############

        ##############
        # calculate the face identification rank 1 to 10 accuracy under the settings of meface face identification
        full_indexes = torch.arange(len(gt_labels))
        for i in range(len(gt_labels)):
            tmp_id = gt_labels[i]
            eq_mask = torch.eq(gt_labels, tmp_id)
            eq_mask[i] = torch.tensor(0).bool()
            eq_indexes = full_indexes[eq_mask]
            ne_mask = torch.ne(gt_labels, tmp_id)
            ne_indexes = full_indexes[ne_mask]
            tmp_probe_set = torch.index_select(
                total_xs_embedding, 0, eq_indexes)
            tmp_gallery_set = torch.index_select(
                total_xs_embedding, 0, ne_indexes)
            tmp_gallery_set = torch.cat(
                (tmp_gallery_set, total_xs_embedding[i, :].view(1, -1)), dim=0)
            true_tmp_gallery_labels = torch.cat(
                (gt_labels[ne_mask], torch.tensor([tmp_id])), dim=0).view(1, -1)
            true_tmp_gallery_labels = true_tmp_gallery_labels.expand(
                tmp_probe_set.size(0), -1)
            tmp_logits = tmp_probe_set.mm(tmp_gallery_set.t())
            _, pred_poses = tmp_logits.topk(
                10, dim=1, largest=True, sorted=True)
            pred_tmp_gallery_labels_ranks = torch.zeros_like(pred_poses)

            for j in range(tmp_logits.size(0)):
                pred_tmp_gallery_labels_ranks[j, :] = true_tmp_gallery_labels[j, :].index_select(
                    0, pred_poses[j, :])
            tmp_probe_labels_ranks = torch.full_like(
                pred_tmp_gallery_labels_ranks, tmp_id)

            if i == 0:
               # total_probe_labels_ranks = copy.deepcopy(tmp_probe_labels_ranks)
               # total_pred_gallery_labels_ranks = copy.deepcopy(pred_tmp_gallery_labels_ranks)
                total_probe_labels_ranks = tmp_probe_labels_ranks
                total_pred_gallery_labels_ranks = pred_tmp_gallery_labels_ranks
            else:
                total_probe_labels_ranks = torch.cat(
                    (total_probe_labels_ranks, tmp_probe_labels_ranks), dim=0)
                total_pred_gallery_labels_ranks = torch.cat(
                    (total_pred_gallery_labels_ranks, pred_tmp_gallery_labels_ranks), dim=0)

        correct = total_probe_labels_ranks.eq(total_pred_gallery_labels_ranks)
        id_acces = []
        for k in range(1, 11):
            correct_k = correct[:,
                                0:k].reshape(-1).float().sum(0, keepdim=True)
            id_acces.append(correct_k.mul_(
                100.0 / total_probe_labels_ranks.size(0)).item())
        ##############
        with open(log_file, 'a+') as f:
            f.write('Test on FGNET dataset, --{} mix-- megafce_id_acces:{:.2f}%, LOPO_id_acc:{:.2f}%, '
                    'argmax_MAE:{:.2f}, mi_estimation:{:.2f}, loglikelihood:{:.2f}\n'.format(
                        crops_mix_mode, id_acces[0], fr_acc_LOPO, argmax_MAE.avg, mi_est.avg, loglikeli.avg))


def test_face_verification(test_loaders, model, test_set_name, log_file, args, crop_number=2, crops_mix_mode='concatenate'):
    model.train(False)
    with torch.no_grad():
        loader_list = [iter(test_loaders[i]) for i in range(crop_number)]
        for j in range(len(loader_list[0])):
            for k in range(crop_number):
                xs_a, xs_b, mb_lbls = loader_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs_a = xs_a.cuda(args.gpu, non_blocking=True)
                    xs_b = xs_b.cuda(args.gpu, non_blocking=True)
                xs_a_embedding, _, _, _ = model(xs_a, emb=True)
                xs_b_embedding, _, _, _ = model(xs_b, emb=True)
                if k == 0:
                    concatenate_xs_a_embedding = xs_a_embedding
                    concatenate_xs_b_emdedding = xs_b_embedding
                else:
                    if crops_mix_mode == 'concatenate':
                        concatenate_xs_a_embedding = torch.cat(
                            (concatenate_xs_a_embedding, xs_a_embedding), dim=1)
                        concatenate_xs_b_emdedding = torch.cat(
                            (concatenate_xs_b_emdedding, xs_b_embedding), dim=1)
                    else:
                        concatenate_xs_a_embedding = torch.add(
                            concatenate_xs_a_embedding, xs_a_embedding)
                        concatenate_xs_b_emdedding = torch.add(
                            concatenate_xs_b_emdedding, xs_b_embedding)
            if j == 0:
                total_xs_a_embedding = concatenate_xs_a_embedding
                total_xs_b_embedding = concatenate_xs_b_emdedding
                gt_labels = mb_lbls
            else:
                total_xs_a_embedding = torch.cat(
                    (total_xs_a_embedding, concatenate_xs_a_embedding), dim=0)
                total_xs_b_embedding = torch.cat(
                    (total_xs_b_embedding, concatenate_xs_b_emdedding), dim=0)
                gt_labels = torch.cat((gt_labels, mb_lbls), dim=0)

        total_xs_a_embedding = F.normalize(
            total_xs_a_embedding.cpu(), p=2, dim=1)
        total_xs_b_embedding = F.normalize(
            total_xs_b_embedding.cpu(), p=2, dim=1)
        total_xs_a_embedding = total_xs_a_embedding.numpy()
        total_xs_b_embedding = total_xs_b_embedding.numpy()
        gt_labels = gt_labels.numpy().astype(bool)
        thresholds = np.arange(-1, 1.01, 0.01)
        _, _, auc, eer, verif_acc, _ = calculate_roc_auc_eer(
            thresholds,
            total_xs_a_embedding,
            total_xs_b_embedding,
            gt_labels,
            nrof_folds=10)
        with open(log_file, 'a+') as f:
            f.write('Test on {} dataset --{} mix-- AUC:{:.2f}%, EER:{:.2f}%, Verif_accuracy:{:.2f}%\n'.format(
                test_set_name, crops_mix_mode, auc*100.0, eer*100.0, verif_acc*100.0))


def test_face_identification_morph2(test_loaders_probe_3000, test_loaders_gallery_3000,
                                    test_loaders_probe_10000, test_loaders_gallery_10000,
                                    model, log_file, args, crop_number=2, crops_mix_mode='concatenate'):
    model.train(False)
    with torch.no_grad():
        test_loaders_probe_3000_list = [
            iter(test_loaders_probe_3000[i]) for i in range(crop_number)]
        for j in range(len(test_loaders_probe_3000_list[0])):
            for k in range(crop_number):
                xs, ys, _ = test_loaders_probe_3000_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs = xs.cuda(args.gpu, non_blocking=True)
                bs_xs_embedding, _, _, _ = model(xs, emb=True)
                if k == 0:
                    mixed_bs_xs_embedding = bs_xs_embedding
                else:
                    if crops_mix_mode == 'concatenate':
                        mixed_bs_xs_embedding = torch.cat(
                            (mixed_bs_xs_embedding, bs_xs_embedding), dim=1)
                    else:
                        mixed_bs_xs_embedding = torch.add(
                            mixed_bs_xs_embedding, bs_xs_embedding)
            if j == 0:
                total_xs_embedding = mixed_bs_xs_embedding
                gt_labels = ys
            else:
                total_xs_embedding = torch.cat(
                    (total_xs_embedding, mixed_bs_xs_embedding), dim=0)
                gt_labels = torch.cat((gt_labels, ys), dim=0)
        total_xs_embedding_probe_3000 = F.normalize(
            total_xs_embedding.cpu(), p=2, dim=1)
        gt_labels_probe_3000 = gt_labels
        ########################
        test_loaders_gallery_3000_list = [
            iter(test_loaders_gallery_3000[i]) for i in range(crop_number)]
        for j in range(len(test_loaders_gallery_3000_list[0])):
            for k in range(crop_number):
                xs, ys, _ = test_loaders_gallery_3000_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs = xs.cuda(args.gpu, non_blocking=True)
                bs_xs_embedding, _, _, _ = model(xs, emb=True)
                if k == 0:
                    mixed_bs_xs_embedding = bs_xs_embedding
                else:
                    if crops_mix_mode == 'concatenate':
                        mixed_bs_xs_embedding = torch.cat(
                            (mixed_bs_xs_embedding, bs_xs_embedding), dim=1)
                    else:
                        mixed_bs_xs_embedding = torch.add(
                            mixed_bs_xs_embedding, bs_xs_embedding)
            if j == 0:
                total_xs_embedding = mixed_bs_xs_embedding
                gt_labels = ys
            else:
                total_xs_embedding = torch.cat(
                    (total_xs_embedding, mixed_bs_xs_embedding), dim=0)
                gt_labels = torch.cat((gt_labels, ys), dim=0)
        total_xs_embedding_gallery_3000 = F.normalize(
            total_xs_embedding.cpu(), p=2, dim=1)
        gt_labels_gallery_3000 = gt_labels
        ########################
        test_loaders_probe_10000_list = [
            iter(test_loaders_probe_10000[i]) for i in range(crop_number)]
        for j in range(len(test_loaders_probe_10000_list[0])):
            for k in range(crop_number):
                xs, ys, _ = test_loaders_probe_10000_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs = xs.cuda(args.gpu, non_blocking=True)
                bs_xs_embedding, _, _, _ = model(xs, emb=True)
                if k == 0:
                    mixed_bs_xs_embedding = bs_xs_embedding
                else:
                    if crops_mix_mode == 'concatenate':
                        mixed_bs_xs_embedding = torch.cat(
                            (mixed_bs_xs_embedding, bs_xs_embedding), dim=1)
                    else:
                        mixed_bs_xs_embedding = torch.add(
                            mixed_bs_xs_embedding, bs_xs_embedding)
            if j == 0:
                total_xs_embedding = mixed_bs_xs_embedding
                gt_labels = ys
            else:
                total_xs_embedding = torch.cat(
                    (total_xs_embedding, mixed_bs_xs_embedding), dim=0)
                gt_labels = torch.cat((gt_labels, ys), dim=0)
        total_xs_embedding_probe_10000 = F.normalize(
            total_xs_embedding.cpu(), p=2, dim=1)
        gt_labels_probe_10000 = gt_labels
        ########################
        test_loaders_gallery_10000_list = [
            iter(test_loaders_gallery_10000[i]) for i in range(crop_number)]
        for j in range(len(test_loaders_gallery_10000_list[0])):
            for k in range(crop_number):
                xs, ys, _ = test_loaders_gallery_10000_list[k].next()
                if args.gpu is not None and torch.cuda.is_available():
                    xs = xs.cuda(args.gpu, non_blocking=True)
                bs_xs_embedding, _, _, _ = model(xs, emb=True)
                if k == 0:
                    mixed_bs_xs_embedding = bs_xs_embedding
                else:
                    if crops_mix_mode == 'concatenate':
                        mixed_bs_xs_embedding = torch.cat(
                            (mixed_bs_xs_embedding, bs_xs_embedding), dim=1)
                    else:
                        mixed_bs_xs_embedding = torch.add(
                            mixed_bs_xs_embedding, bs_xs_embedding)
            if j == 0:
                total_xs_embedding = mixed_bs_xs_embedding
                gt_labels = ys
            else:
                total_xs_embedding = torch.cat(
                    (total_xs_embedding, mixed_bs_xs_embedding), dim=0)
                gt_labels = torch.cat((gt_labels, ys), dim=0)
        total_xs_embedding_gallery_10000 = F.normalize(
            total_xs_embedding.cpu(), p=2, dim=1)
        gt_labels_gallery_10000 = gt_labels
        ########################

        # calculate the face identification rank-1 accuracy
        logits_3000 = total_xs_embedding_probe_3000.mm(
            total_xs_embedding_gallery_3000.t())
        _, pred_pos_3000 = logits_3000.topk(
            1, dim=1, largest=True, sorted=True)
        pred_pos_3000 = pred_pos_3000.squeeze()
        pred_labels_3000 = torch.zeros_like(gt_labels_gallery_3000)
        for i in range(len(pred_labels_3000)):
            pred_labels_3000[i] = gt_labels_gallery_3000[pred_pos_3000[i]]
        setting3000_id_rank1_acc = accuracy_percent(
            gt_labels_probe_3000, pred_labels_3000)

        logits_10000 = total_xs_embedding_probe_10000.mm(
            total_xs_embedding_gallery_10000.t())
        _, pred_pos_10000 = logits_10000.topk(
            1, dim=1, largest=True, sorted=True)
        pred_pos_10000 = pred_pos_10000.squeeze()
        pred_labels_10000 = torch.zeros_like(gt_labels_gallery_10000)
        for i in range(len(pred_labels_10000)):
            pred_labels_10000[i] = gt_labels_gallery_10000[pred_pos_10000[i]]
        setting10000_id_rank1_acc = accuracy_percent(
            gt_labels_probe_10000, pred_labels_10000)
        #############
        with open(log_file, 'a+') as f:
            f.write('Test on MORPH2 dataset --{} mix-- setting3000_id_rank1_acc:{:.2f}%, '
                    'setting10000_id_rank1_acc:{:.2f}%\n'.format(
                        crops_mix_mode, setting3000_id_rank1_acc, setting10000_id_rank1_acc))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
