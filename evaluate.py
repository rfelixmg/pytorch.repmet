from __future__ import print_function

import os
import argparse

import torch.backends.cudnn as cudnn

import configs
from utils import *
from data.load import load_datasets
from models.load import load_net

assert torch.cuda.is_available(), 'Error: CUDA not found!'


def evaluate(run_id,
             set_name,
             model_name,
             chunk_size=32,
             split='test',
             load_iteration=-1,
             load_path=configs.general.paths.models,
             plots_path=configs.general.paths.graphing,
             plots_ext='.png'):

    # Setup load path
    load_path = os.path.join(load_path, "%s" % run_id)

    # Setup plotting directory
    plots_path = os.path.join(plots_path, "%s" % run_id)
    os.makedirs(plots_path, exist_ok=True)

    net, input_size = load_net(model_name)

    # Load set and get train and test labels from datasets
    train_dataset, test_dataset = load_datasets(set_name, input_size=input_size)
    if split == 'train':
        dataset = train_dataset
    else:
        dataset = test_dataset
    y = get_labels(dataset)

    # Use the GPU
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    cudnn.benchmark = True

    # Create loss object (this stores the cluster centroids)
    if load_iteration < 0:
        l = os.listdir(load_path)
        l.sort(reverse=True)
        state = torch.load("%s/%s" % (load_path, l[1])) # ignore log.txt
        print("Loading model: %s/%s" % (load_path, l[1]))
    else:
        state = torch.load("%s/i%06d%s" % (load_path, load_iteration, '.pth')) # ignore log.txt
        print("%s/i%06d%s" % (load_path, load_iteration, '.pth'))

    net.load_state_dict(state['state_dict'])

    the_loss = state['the_loss']
    plot_classes = state['plot_classes']

    cluster_indexs = []
    for ci in range(len(the_loss.cluster_classes)):
        if the_loss.cluster_classes[ci] in plot_classes:
            cluster_indexs.append(ci)

    # calc all the accs
    x = compute_reps(net, test_dataset, list(range(len(y))), chunk_size=chunk_size)

    test_acc = the_loss.calc_accuracy(x, y, method='simple')
    test_acc_b = the_loss.calc_accuracy(x, y, method='magnet')
    test_acc_c = the_loss.calc_accuracy(x, y, method='repmet')
    test_acc_d = the_loss.calc_accuracy(x, y, method='unsupervised')

    print("simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f" % (test_acc, test_acc_b, test_acc_c, test_acc_d))

    graph(x, y,
          cluster_centers=ensure_numpy(the_loss.centroids),
          cluster_classes=the_loss.cluster_classes,
          savepath="%s/test-%s%s" % (plots_path, split, plots_ext))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DML Evaluation')
    parser.add_argument('--run_id', required=True, help='experiment run name', default='000')
    parser.add_argument('--set_name', required=True, help='dataset name', default='mnist')
    parser.add_argument('--model_name', required=True, help='model name', default='mnist_default')
    parser.add_argument('--chunk_size', required=False, help='the chunk/batch size for calculating embeddings (lower for less mem)', default=32, type=int)
    parser.add_argument('--split', required=False, help='train/test', default='test')
    parser.add_argument('--load_iteration', required=False, help='load this iteration (-1 will be latest)', default=-1)
    parser.add_argument('--load_path', required=False, help='where to load the models from', default=configs.general.paths.models)
    parser.add_argument('--plots_path', required=False, help='where to save the plots', default=configs.general.paths.graphing)
    parser.add_argument('--plots_ext', required=False, help='.png/.pdf', default='.png')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # args = parse_args()
    # evaluate(run_id=args.run_id,
    #          set_name=args.set_name,
    #          model_name=args.model_name,
    #          chunk_size=args.chunk_size,
    #          split=args.split,
    #          load_iteration=args.load_iteration,
    #          load_path=args.load_path,
    #          plots_path=args.plots_path,
    #          plots_ext=args.plots_ext,
    #          n_plot_samples=args.n_plot_samples)

    evaluate('testml_nonsqr', 'oxford_flowers', 'resnet18_e1024', split='train')