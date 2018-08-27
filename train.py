from __future__ import print_function

import os
import argparse

import torch.backends.cudnn as cudnn

import configs
from utils import *
from magnet_loss import MagnetLoss
from repmet_loss import RepMetLoss, RepMetLoss2, RepMetLoss3
from my_loss import MyLoss1
from data.load import load_datasets
from models.load import load_net

assert torch.cuda.is_available(), 'Error: CUDA not found!'


def train(run_id,
          set_name,
          model_name,
          loss_type,
          m, d, k, alpha,
          n_iterations=0000,
          net_learning_rate=0.0001,
          cluster_learning_rate=0.001,
          chunk_size=32,
          refresh_clusters=50,
          norm_clusters=False,
          calc_acc_every=100,
          load_latest=True,
          save_every=1000,
          save_path=configs.general.paths.models,
          plot_every=500,
          plots_path=configs.general.paths.graphing,
          plots_ext='.png',
          n_plot_samples=10,
          n_plot_classes=10):


    # Setup model directory
    save_path = os.path.join(save_path, "%s" % run_id)
    os.makedirs(save_path, exist_ok=True)

    # Setup plotting directory
    plots_path = os.path.join(plots_path, "%s" % run_id)
    os.makedirs(plots_path, exist_ok=True)

    net, input_size = load_net(model_name)

    # Load set and get train and test labels from datasets
    train_dataset, test_dataset = load_datasets(set_name, input_size=input_size)#299 for inception
    train_y = get_labels(train_dataset)
    test_y = get_labels(test_dataset)

    # Use the GPU
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    cudnn.benchmark = True

    # make list of cluster refresh if given an interval int
    if isinstance(refresh_clusters, int):
        refresh_clusters = list(range(0, n_iterations, refresh_clusters))

    # Get initial embedding using all samples in training set
    initial_reps = compute_all_reps(net, train_dataset, chunk_size)

    # Create loss object (this stores the cluster centroids)
    if loss_type == "magnet":
        the_loss = MagnetLoss(train_y, k, m, d, alpha=alpha)

        # Initialise the embeddings/representations/clusters
        print("Initialising the clusters")
        the_loss.update_clusters(initial_reps)

        # Setup the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net_learning_rate)
        optimizerb = None
    elif loss_type == "repmet" or loss_type == "repmet2" or loss_type == "repmet3" or loss_type == "myloss1":
        if loss_type == "repmet":
            the_loss = RepMetLoss(train_y, k, m, d, alpha=alpha)
        elif loss_type == "repmet2":
            the_loss = RepMetLoss2(train_y, k, m, d, alpha=alpha)
        elif loss_type == "repmet3":
            the_loss = RepMetLoss3(train_y, k, m, d, alpha=alpha)
        elif loss_type == "myloss1":
            the_loss = MyLoss1(train_y, k, m, d, alpha=alpha)

        # Initialise the embeddings/representations/clusters
        print("Initialising the clusters")
        the_loss.update_clusters(initial_reps)

        # Setup the optimizer
        if cluster_learning_rate < 0:
            optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())) + [the_loss.centroids], lr=net_learning_rate)
            optimizerb = None
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=net_learning_rate)
            optimizerb = torch.optim.Adam([the_loss.centroids], lr=cluster_learning_rate)

    l = os.listdir(save_path)
    if load_latest and len(l) > 1:
        l.sort(reverse=True)
        state = torch.load("%s/%s" % (save_path, l[1])) # ignore log.txt

        print("Loading model: %s/%s" % (save_path, l[1]))

        net.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        if optimizerb:
            optimizerb.load_state_dict(state['optimizerb'])

        start_iteration = state['iteration']+1
        best_acc = state['best_acc']
        the_loss = state['the_loss'] # overwrite the loss
        plot_sample_indexs = state['plot_sample_indexs']
        plot_classes = state['plot_classes']
        plot_test_sample_indexs = state['plot_test_sample_indexs']
        plot_test_classes = state['plot_test_classes']
        batch_losses = state['batch_losses']
        train_accs = state['train_accs']
        test_accs = state['test_accs']

        test_acc = test_accs[0][-1]
        train_acc = train_accs[0][-1]
        test_acc_b = test_accs[1][-1]
        train_acc_b = train_accs[1][-1]
        test_acc_c = test_accs[2][-1]
        train_acc_c = train_accs[2][-1]
        test_acc_d = test_accs[3][-1]
        train_acc_d = train_accs[3][-1]
    else:

        # Randomly sample the classes then the samples from each class to plot
        plot_sample_indexs, plot_classes = get_indexs(train_y, n_plot_classes, n_plot_samples)
        plot_test_sample_indexs, plot_test_classes = get_indexs(test_y, n_plot_classes, n_plot_samples, class_ids=plot_classes)

        batch_losses = []
        train_accs = [[], [], [], []]
        test_accs = [[], [], [], []]
        start_iteration = 0
        best_acc = 0
        test_acc = 0
        train_acc = 0
        test_acc_b = 0
        train_acc_b = 0
        test_acc_c = 0
        train_acc_c = 0
        test_acc_d = 0
        train_acc_d = 0

    # lets plot the initial embeddings
    cluster_classes = the_loss.cluster_classes

    # use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
    for i in range(len(cluster_classes)):
        cluster_classes[i] = the_loss.unique_y[cluster_classes[i]]

    cluster_indexs = []
    for ci in range(len(the_loss.cluster_classes)):
        if the_loss.cluster_classes[ci] in plot_classes:
            cluster_indexs.append(ci)

    if not load_latest or len(l) < 2:
        # plot it
        graph(initial_reps[plot_sample_indexs], train_y[plot_sample_indexs],
              cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
              cluster_classes=the_loss.cluster_classes[cluster_indexs],
              savepath="%s/emb-initial%s" % (plots_path, plots_ext))

    # Get some sample indxs to do acc test on... compare these to the acc coming out of the batch calc
    test_train_inds,_ = get_indexs(train_y, len(set(train_y)), 10)

    # Lets setup the training loop
    iteration = None
    for iteration in range(start_iteration, n_iterations):
        # Sample batch and do forward-backward
        batch_example_inds, batch_class_inds = the_loss.gen_batch()

        # Get inputs and and labels from the dataset
        batch_x = get_inputs(train_dataset, batch_example_inds).cuda()
        batch_y = torch.from_numpy(batch_class_inds).cuda()

        # Calc the outputs (embs) and then the loss + accs
        outputs = net(batch_x)
        batch_loss, batch_example_losses, batch_acc = the_loss.loss(outputs, batch_y)

        # Pass the gradient and update
        optimizer.zero_grad()
        if optimizerb:
            optimizerb.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if optimizerb:
            optimizerb.step()

            if norm_clusters:
                # Let's also normalise those centroids [because repmet pushes them away from unit sphere] to:
                # Option 1: sit on the hypersphere (use norm)
                # g = the_loss.centroids.norm(p=2,dim=0,keepdim=True)
                import torch.nn.functional as F
                the_loss.centroids.data = F.normalize(the_loss.centroids)

                # Option 2: sit on OR within the hypersphere (divide by max [scales all evenly]))
                # mx, _ = the_loss.centroids.max(0)
                # mx, _ = mx.max(0)
                # the_loss.centroids.data = the_loss.centroids/mx
                # What you wrote here doesn't work as scales axes independently...

        # Just changing some types
        batch_loss = float(ensure_numpy(batch_loss))
        batch_example_losses = ensure_numpy(batch_example_losses)

        # Update loss index
        the_loss.update_losses(batch_example_inds, batch_example_losses)

        if iteration > 0 and not iteration % calc_acc_every:
            # calc all the accs
            train_reps = compute_reps(net, train_dataset, test_train_inds, chunk_size)
            test_test_inds, _ = get_indexs(test_y, len(set(test_y)), 10)
            test_reps = compute_reps(net, test_dataset, test_test_inds, chunk_size)

            test_acc = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='simple')
            train_acc = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='simple')

            test_acc_b = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='magnet')
            train_acc_b = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='magnet')

            test_acc_c = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='repmet')
            train_acc_c = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='repmet')

            # removed because of failed runs with out of mem errors
            # test_acc_d = the_loss.calc_accuracy(test_reps, test_y[test_test_inds], method='unsupervised')
            # train_acc_d = the_loss.calc_accuracy(train_reps, train_y[test_train_inds], method='unsupervised')

            test_acc_d = test_acc_c
            train_acc_d = train_acc_c

            with open(save_path+'/log.txt', 'a') as f:
                f.write("Iteration %06d/%06d: Tr. L: %0.3f :: Batch. A: %0.3f :::: Tr. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f :::: Te. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f\n" % (iteration, n_iterations, batch_loss, batch_acc, train_acc, train_acc_b, train_acc_c, train_acc_d, test_acc, test_acc_b, test_acc_c, test_acc_d))
            print("Iteration %06d/%06d: Tr. L: %0.3f :: Batch. A: %0.3f :::: Tr. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f :::: Te. A - simple: %0.3f -- magnet: %0.3f -- repmet: %0.3f -- unsupervised: %0.3f" % (iteration, n_iterations, batch_loss, batch_acc, train_acc, train_acc_b, train_acc_c, train_acc_d, test_acc, test_acc_b, test_acc_c, test_acc_d))

            batch_ass_ids = np.unique(the_loss.assignments[batch_example_inds])

            os.makedirs("%s/batch-emb/" % plots_path, exist_ok=True)
            os.makedirs("%s/batch-emb-all/" % plots_path, exist_ok=True)
            os.makedirs("%s/batch-clusters/" % plots_path, exist_ok=True)

            graph(ensure_numpy(outputs),
                  train_y[batch_example_inds],
                  cluster_centers=ensure_numpy(the_loss.centroids)[batch_ass_ids],
                  cluster_classes=the_loss.cluster_classes[batch_ass_ids],
                  savepath="%s/batch-emb/i%06d%s" % (plots_path, iteration, plots_ext))

            graph(ensure_numpy(outputs),
                  train_y[batch_example_inds],
                  cluster_centers=ensure_numpy(the_loss.centroids),
                  cluster_classes=the_loss.cluster_classes,
                  savepath="%s/batch-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))

            graph(np.zeros_like(ensure_numpy(outputs)),
                  np.zeros_like(train_y[batch_example_inds]),
                  cluster_centers=ensure_numpy(the_loss.centroids),
                  cluster_classes=the_loss.cluster_classes,
                  savepath="%s/batch-clusters/i%06d%s" % (plots_path, iteration, plots_ext))

        train_reps_this_iter = False
        if iteration in refresh_clusters:
            with open(save_path+'/log.txt', 'a') as f:
                f.write('Refreshing clusters')
            print('Refreshing clusters')
            train_reps = compute_all_reps(net, train_dataset, chunk_size=chunk_size)
            the_loss.update_clusters(train_reps)

            cluster_classes = the_loss.cluster_classes
            train_reps_this_iter = True

        # store the stats to graph at end
        batch_losses.append(batch_loss)
        # batch_accs.append(batch_acc)
        train_accs[0].append(train_acc)
        test_accs[0].append(test_acc)
        train_accs[1].append(train_acc_b)
        test_accs[1].append(test_acc_b)
        train_accs[2].append(train_acc_c)
        test_accs[2].append(test_acc_c)
        train_accs[3].append(train_acc_d)
        test_accs[3].append(test_acc_d)

        if iteration > 0 and not iteration % plot_every:
            #use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
            for i in range(len(cluster_classes)):
                cluster_classes[i] = the_loss.unique_y[cluster_classes[i]]

            # so 1. we don't have to recalc, 2. the kmeans update occured on these reps, better graphing ...
            # if we were to re-get with compute_reps(), batch norm and transforms could give different embeddings
            if train_reps_this_iter:
                plot_train_emb = train_reps[test_train_inds]
            else:
                plot_train_emb = compute_reps(net, train_dataset, test_train_inds, chunk_size=chunk_size)

            plot_test_emb = compute_reps(net, test_dataset, plot_test_sample_indexs, chunk_size=chunk_size)

            os.makedirs("%s/train-emb/" % plots_path, exist_ok=True)
            os.makedirs("%s/test-emb/" % plots_path, exist_ok=True)
            os.makedirs("%s/train-emb-all/" % plots_path, exist_ok=True)
            os.makedirs("%s/test-emb-all/" % plots_path, exist_ok=True)
            os.makedirs("%s/cluster-losses/" % plots_path, exist_ok=True)
            os.makedirs("%s/cluster-counts/" % plots_path, exist_ok=True)

            graph(plot_train_emb,
                  train_y[plot_sample_indexs],
                  cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
                  cluster_classes=the_loss.cluster_classes[cluster_indexs],
                  savepath="%s/train-emb/i%06d%s" % (plots_path, iteration, plots_ext))

            graph(plot_test_emb,
                  test_y[plot_test_sample_indexs],
                  cluster_centers=ensure_numpy(the_loss.centroids)[cluster_indexs],
                  cluster_classes=the_loss.cluster_classes[cluster_indexs],
                  savepath="%s/test-emb/i%06d%s" % (plots_path, iteration, plots_ext))

            graph(plot_train_emb,
                  # train_y[plot_sample_indexs],
                  train_y[test_train_inds],
                  cluster_centers=ensure_numpy(the_loss.centroids),
                  cluster_classes=the_loss.cluster_classes,
                  savepath="%s/train-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))

            graph(plot_test_emb,
                  test_y[plot_test_sample_indexs],
                  cluster_centers=ensure_numpy(the_loss.centroids),
                  cluster_classes=the_loss.cluster_classes,
                  savepath="%s/test-emb-all/i%06d%s" % (plots_path, iteration, plots_ext))

            plot_smooth({'loss': batch_losses,
                         'train acc': train_accs[0],
                         'test acc': test_accs[0]},
                        savepath="%s/loss_simple%s" % (plots_path, plots_ext))
            plot_smooth({'loss': batch_losses,
                         'train acc': train_accs[1],
                         'test acc': test_accs[1]},
                        savepath="%s/loss_magnet%s" % (plots_path, plots_ext))
            plot_smooth({'loss': batch_losses,
                         'train acc': train_accs[2],
                         'test acc': test_accs[2]},
                        savepath="%s/loss_repmet%s" % (plots_path, plots_ext))
            # plot_smooth({'loss': batch_losses,
            #              'train acc': train_accs[3],
            #              'test acc': test_accs[3]},
            #             savepath="%s/loss_unsupervised%s" % (plots_path, plots_ext))

            plot_cluster_data(the_loss.cluster_losses,
                              the_loss.cluster_classes,
                              title="cluster losses",
                              savepath="%s/cluster-losses/i%06d%s" % (plots_path, iteration, plots_ext))

            cluster_counts = []
            for c in range(len(the_loss.cluster_assignments)):
                cluster_counts.append(len(the_loss.cluster_assignments[c]))

            plot_cluster_data(cluster_counts,
                              the_loss.cluster_classes,
                              title="cluster counts",
                              savepath="%s/cluster-counts/i%06d%s" % (plots_path, iteration, plots_ext))

        if iteration > 0 and not iteration % save_every:
            if save_path:
                if test_acc_d > best_acc:
                    print("Saving model (is best): %s/i%06d%s" % (save_path, iteration, '.pth'))
                    best_acc = test_acc_d
                else:
                    print("Saving model: %s/i%06d%s" % (save_path, iteration, '.pth'))

                state = {
                    'iteration': iteration,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': test_acc_d,
                    'best_acc': best_acc,
                    'the_loss': the_loss,
                    'plot_sample_indexs': plot_sample_indexs,
                    'plot_classes': plot_classes,
                    'plot_test_sample_indexs': plot_test_sample_indexs,
                    'plot_test_classes': plot_test_classes,
                    'batch_losses': batch_losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs,
                }
                if optimizerb:
                    state['optimizerb'] = optimizerb.state_dict()
                torch.save(state, "%s/i%06d%s" % (save_path, iteration, '.pth'))

    # END TRAINING LOOP

    # Plot curves and graphs
    plot_smooth({'loss': batch_losses,
                 'train acc': train_accs[0],
                 'test acc': test_accs[0]},
                savepath="%s/loss_simple%s" % (plots_path, plots_ext))
    plot_smooth({'loss': batch_losses,
                 'train acc': train_accs[1],
                 'test acc': test_accs[1]},
                savepath="%s/loss_magnet%s" % (plots_path, plots_ext))
    plot_smooth({'loss': batch_losses,
                 'train acc': train_accs[2],
                 'test acc': test_accs[2]},
                savepath="%s/loss_repmet%s" % (plots_path, plots_ext))
    plot_smooth({'loss': batch_losses,
                 'train acc': train_accs[3],
                 'test acc': test_accs[3]},
                savepath="%s/loss_unsupervised%s" % (plots_path, plots_ext))

    # Calculate and graph the final
    final_reps = compute_reps(net, train_dataset, plot_sample_indexs, chunk_size=chunk_size)
    graph(final_reps, train_y[plot_sample_indexs], savepath="%s/emb-final%s" % (plots_path, plots_ext))

    if save_path and iteration:
        if test_acc_d > best_acc:
            print("Saving model (is best): %s/i%06d%s" % (save_path, iteration, '.pth'))
            best_acc = test_acc_d
        else:
            print("Saving model: %s/i%06d%s" % (save_path, iteration, '.pth'))

        state = {
            'iteration': iteration,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': test_acc_d,
            'best_acc': best_acc,
            'the_loss': the_loss,
            'plot_sample_indexs': plot_sample_indexs,
            'plot_classes': plot_classes,
            'plot_test_sample_indexs': plot_test_sample_indexs,
            'plot_test_classes': plot_test_classes,
            'batch_losses': batch_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
        }
        if optimizerb:
            state['optimizerb'] = optimizerb.state_dict()
        torch.save(state, "%s/i%06d%s" % (save_path, iteration, '.pth'))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DML Training')
    parser.add_argument('--run_id', required=True, help='experiment run name', default='000')
    parser.add_argument('--set_name', required=True, help='dataset name', default='mnist')
    parser.add_argument('--model_name', required=True, help='model name', default='mnist_default')
    parser.add_argument('--loss_type', required=True, help='magnet, repmet, repmet2', default='repmet2')
    parser.add_argument('--m', required=True, help='number of clusters per batch', default=8, type=int)
    parser.add_argument('--d', required=True, help='number of samples per cluster per batch', default=8, type=int)
    parser.add_argument('--k', required=True, help='number of clusters per class', default=3, type=int)
    parser.add_argument('--alpha', required=True, help='cluster margin', default=1.0, type=int)
    parser.add_argument('--n_iterations', required=False, help='number of iterations to perform', default=5000, type=int)
    parser.add_argument('--net_learning_rate', required=False, help='the learning rate for the net', default=0.0001, type=float)
    parser.add_argument('--cluster_learning_rate', required=False, help='the learning rate for the modes (centroids), if -1 will use single optimiser for both net and modes', default=0.001, type=float)
    parser.add_argument('--chunk_size', required=False, help='the chunk/batch size for calculating embeddings (lower for less mem)', default=32, type=int)
    parser.add_argument('--refresh_clusters', required=False, help='refresh the clusters every ? iterations or on these iterations (int or list or ints)', default=[0,1,2])
    parser.add_argument('--calc_acc_every', required=False, help='calculate the accuracy every ? iterations', default=10, type=int)
    parser.add_argument('--load_latest', required=False, help='load a model if presaved', default=True)
    parser.add_argument('--save_every', required=False, help='save the model every ? iterations', default=500, type=int)
    parser.add_argument('--save_path', required=False, help='where to save the models', default=configs.general.paths.models)
    parser.add_argument('--plot_every', required=False, help='plot graphs every ? iterations', default=10, type=int)
    parser.add_argument('--plots_path', required=False, help='where to save the plots', default=configs.general.paths.graphing)
    parser.add_argument('--plots_ext', required=False, help='.png/.pdf', default='.png')
    parser.add_argument('--n_plot_samples', required=False, help='plot ? samples per class', default=10, type=int)
    parser.add_argument('--n_plot_classes', required=False, help='plot ? classes', default=10, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # args = parse_args()
    # train(run_id=args.run_id,
    #       set_name=args.set_name,
    #       model_name=args.model_name,
    #       loss_type=args.loss_type,
    #       m=args.m, d=args.d, k=args.k, alpha=args.alpha,
    #       n_iterations=args.n_iterations,
    #       net_learning_rate=args.net_learning_rate,
    #       cluster_learning_rate=args.cluster_learning_rate,
    #       chunk_size=args.chunk_size,
    #       refresh_clusters=args.refresh_clusters,
    #       calc_acc_every=args.calc_acc_every,
    #       load_latest=args.load_latest,
    #       save_every=args.save_every,
    #       save_path=args.save_path,
    #       plot_every=args.plot_every,
    #       plots_path=args.plots_path,
    #       plots_ext=args.plots_ext,
    #       n_plot_samples=args.n_plot_samples,
    #       n_plot_classes=args.n_plot_classes)

    # MNIST Experiments
    # train('001_r100_k1', 'mnist', 'mnist_default', 'magnet', m=8, d=8, k=1, alpha=1.0, refresh_clusters=100, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('002_r100_k1', 'mnist', 'mnist_default', 'repmet', m=8, d=8, k=1, alpha=1.0, refresh_clusters=100, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('002_nr_k1', 'mnist', 'mnist_default', 'repmet', m=8, d=8, k=1, alpha=1.0, refresh_clusters=2000, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('003_r100_k1', 'mnist', 'mnist_default', 'repmet2', m=8, d=8, k=1, alpha=1.0, refresh_clusters=100, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('003_nr_k1', 'mnist', 'mnist_default', 'repmet2', m=8, d=8, k=1, alpha=1.0, refresh_clusters=2000, calc_acc_every=10, plot_every=10, n_iterations=2000)

    # Flowers Experiments
    # train('004_r50_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_r50_k3_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_r50_k1_inceptionv3_fc2048_e1024_pt_ul_norm', 'oxford_flowers', 'inceptionv3_fc2048_e1024_pt_ul_norm', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)

    # train('006_r50_k1_resnet18_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024', 'repmet2',
    #       m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000,
    #       norm_clusters=True)
    # train('005_r50_k1_resnet18_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024', 'repmet', m=12, d=4, k=1,
    #       alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000, norm_clusters=True)
    # train('005_nr_k1_resnet18_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024', 'repmet', m=12, d=4, k=1,
    #       alpha=1.0, refresh_clusters=5000, calc_acc_every=10, plot_every=10, n_iterations=2000, norm_clusters=True)
    # train('006_nr_k1_resnet18_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024', 'repmet2', m=12, d=4, k=1,
    #       alpha=1.0, refresh_clusters=5000, calc_acc_every=10, plot_every=100, n_iterations=2000, norm_clusters=True)
    #
    # train('005_r50_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('006_r50_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet2', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('005_r1_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=1, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('006_r1_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet2', m=12, d=4, k=1, alpha=1.0, refresh_clusters=1, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('006_r0t2_k3_resnet18_e1024_lr.001_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024', 'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0,1,2], calc_acc_every=10, plot_every=10, n_iterations=2000, norm_clusters=True)
    # train('005_nr_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=5000, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('006_nr_k1_resnet18_e1024', 'oxford_flowers', 'resnet18_e1024', 'repmet2', m=12, d=4, k=1, alpha=1.0, refresh_clusters=5000, calc_acc_every=10, plot_every=10, n_iterations=2000)

    # train('testml_nonsqr', 'oxford_flowers', 'resnet18_e1024',
    #       'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True, save_every=200)
    # train('test', 'oxford_flowers', 'resnet18_e1024',
    #       'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True)
    # train('test2', 'oxford_flowers', 'resnet18_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True, save_every=200)
    # train('test3', 'oxford_flowers', 'resnet18_e1024',
    #       'repmet3', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True, save_every=200)
    train('myloss2', 'oxford_flowers', 'resnet18_e1024',
          'myloss1', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
          n_iterations=1000, norm_clusters=True, save_every=200)

    # train('005_r0t2_k3_resnet18_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet18_e1024',
    #       'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=2000, norm_clusters=True)
    #
    # train('006_r0t2_k3_resnet50_e1024_clust-scaling-norm', 'oxford_flowers', 'resnet50_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=2000, norm_clusters=True)

    # train('006_r0t2_k3_inceptionv3_fc2048_e1024_clust-scaling-norm', 'oxford_flowers', 'inceptionv3_fc2048_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=2000, norm_clusters=True)

    # train('006_r0t2_k3_inceptionv3_fc2048_e1024_ti_clust-scaling-norm', 'oxford_flowers', 'inceptionv3_fc2048_e1024_ti',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=2000, norm_clusters=True, save_every=200)

    # train('009_r0t2_k3_resnet18_e1024_clust-scaling-norm_crp', 'stanford_dogs', 'resnet18_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, save_every=200)
    # train('007_r50_k3_resnet18_e1024_clust-scaling-norm_crp', 'stanford_dogs', 'resnet18_e1024',
    #       'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, save_every=200)

    # train('009_r0t2_k3_resnet50_e1024_clust-scaling-norm_crp', 'stanford_dogs', 'resnet50_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True, save_every=200)
    #
    # train('009_r0t2_k3_inceptionv3_fc2048_e1024_clust-scaling-norm_crp', 'stanford_dogs', 'inceptionv3_fc2048_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, norm_clusters=True, save_every=200)

    # train('007_r50_k3_resnet18_e1024_clust-scaling-norm_crp', 'stanford_dogs', 'resnet18_e1024',
    #       'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, save_every=200, norm_clusters=True)
    #
    # train('009_r0t2_k3_resnet50_e1024_crp', 'stanford_dogs', 'resnet50_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, save_every=200)
    #
    # train('009_r0t2_k3_inceptionv3_fc2048_e1024_crp', 'stanford_dogs', 'inceptionv3_fc2048_e1024',
    #       'repmet2', m=12, d=4, k=3, alpha=1.0, refresh_clusters=[0, 1, 2], calc_acc_every=10, plot_every=10,
    #       n_iterations=1000, save_every=200)

    # train('003_k3', 'mnist', 'mnist_default', 'repmet', m=8, d=8, k=3, alpha=1.0, refresh_clusters=1000, calc_acc_every=10, plot_every=10, n_iterations=1000)
    # train('004_k1', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=1000)
    # train('004_k3', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('005_k1', 'oxford_flowers', 'resnet18_e1024_pt', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('005_k1_nr', 'oxford_flowers', 'resnet18_e1024_pt', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=1000, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('005_k3_nr', 'oxford_flowers', 'resnet18_e1024_pt', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=1000, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('005_k3', 'oxford_flowers', 'resnet18_e1024_pt', 'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('004-10000_r50', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=10000)
    # train('004-10000_k3', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=10000)
    # train('004_del', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=3000)
    # train('004_del_k3', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=3000)
    # train('005_k1', 'oxford_flowers', 'inceptionv3_e1024_pt', 'magnet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=1000)
    # train('004_k1', 'oxford_flowers', 'resnet18_e1024_fc2048_pt', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=1000)

    # train('004_k1_resnet18_e512', 'oxford_flowers', 'resnet18_e512', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e1024_fc2048_norm', 'oxford_flowers', 'resnet18_e1024_fc2048_norm', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e1024_fc2048_norm_pt', 'oxford_flowers', 'resnet18_e1024_fc2048_norm_pt', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e512_fc512', 'oxford_flowers', 'resnet18_e512_fc512', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e1024_fc1024', 'oxford_flowers', 'resnet18_e1024_fc1024', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e1024_pt', 'oxford_flowers', 'resnet18_e1024_pt', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_e1024_fc1024_norm', 'oxford_flowers', 'resnet18_e1024_fc1024_norm', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)

    # train('004_k1_inceptionv3_fc2048_e1024_pt_l', 'oxford_flowers', 'inceptionv3_fc2048_e1024_pt_l', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet50_e1024_fc2048_norm_pt', 'oxford_flowers', 'resnet50_e1024_fc2048_norm_pt', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k3_resnet50_e1024_fc2048_norm_pt', 'oxford_flowers', 'resnet50_e1024_fc2048_norm_pt', 'magnet', m=12, d=4, k=3, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('005_k1_resnet50_e1024_fc2048_norm_pt', 'oxford_flowers', 'resnet50_e1024_fc2048_norm_pt', 'repmet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('005_k3_resnet50_e1024_fc2048_norm_pt', 'oxford_flowers', 'resnet50_e1024_fc2048_norm_pt', 'repmet', m=12, d=4, k=3, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)

    # train('004_k1_resnet18_fc2048_e2_pt_ul', 'oxford_flowers', 'resnet18_fc2048_e2_pt_ul', 'magnet',
    # train('004_k1_resnet18_e1024_pt_norm', 'oxford_flowers', 'resnet18_e1024_pt_norm', 'magnet',
    #       m=12, d=4, k=1, alpha=1.0, refresh_clusters=50, calc_acc_every=10, plot_every=100, n_iterations=2000)
    # train('004_k1_resnet18_fc2048_e2_pt_ul_nr', 'oxford_flowers', 'resnet18_fc2048_e2_pt_ul', 'magnet',
    #       m=12, d=4, k=1, alpha=1.0, refresh_clusters=10, calc_acc_every=10, plot_every=100, n_iterations=1000)

    # train('005_k1_inceptionv3_fc2048_e2_pt_ul_nr', 'oxford_flowers', 'inceptionv3_fc2048_e2_pt_ul', 'repmet2',
    #       m=12, d=4, k=1, alpha=1.0, refresh_clusters=2000, calc_acc_every=10, plot_every=5, n_iterations=500)
    # train('005_k1_inceptionv3_fc2048_e2_pt_ul_r1', 'oxford_flowers', 'inceptionv3_fc2048_e2_pt_ul', 'repmet2',
    #       m=12, d=4, k=1, alpha=1.0, refresh_clusters=1, calc_acc_every=10, plot_every=5, n_iterations=500)

    # train('006_k3_resnet50_e1024_fc2048_norm_pt', 'stanford_dogs', 'resnet50_e1024_fc2048_norm_pt', 'magnet', m=12, d=4, k=3, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('006_k1_resnet50_e1024_fc2048_norm_pt', 'stanford_dogs', 'resnet50_e1024_fc2048_norm_pt', 'magnet', m=12, d=4, k=1, alpha=1.0,
    #       refresh_clusters=50, calc_acc_every=10, plot_every=10, n_iterations=2000)
    # train('003', 'oxford_flowers', 'resnet50_e512', 'magnet', m=12, d=4, k=3, alpha=1.0)
    # train('004', 'oxford_flowers', 'resnet50_e512', 'repmet', m=12, d=4, k=3, alpha=1.0)
    # train('007b', 'oxford_flowers', 'resnet50_e512', 'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=2000)
    # train('007c', 'oxford_flowers', 'resnet50_e1024_fc1024', 'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=5000) # norm=True
    # train('007d', 'oxford_flowers', 'resnet50_e1024_fc1024', 'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=5000) # norm=False
    # train('007e', 'oxford_flowers', 'resnet50_e1024_fc1024', 'repmet', m=12, d=4, k=3, alpha=1.0, refresh_clusters=2000) # cosine dist
    # train('007f', 'oxford_flowers', 'resnet18_e1024_fc1024', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=10, calc_acc_every=5, plot_every=5) # cosine dist
    # train('008f', 'oxford_flowers', 'resnet18_e1024_fc1024', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=10, calc_acc_every=5, plot_every=5) # cosine dist
    # train('008h', 'oxford_flowers', 'resnet18_e1024_fc1024', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=100, calc_acc_every=5, plot_every=5, n_iterations=2000) # cosine dist
    # train('007g', 'oxford_flowers', 'resnet18_e1024_fc1024', 'magnet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=40, calc_acc_every=250, plot_every=500) # cosine dist
    # train('007j', 'oxford_flowers', 'resnet18_e1024_fc2048_pt', 'repmet', m=12, d=4, k=1, alpha=1.0, refresh_clusters=200, calc_acc_every=10, plot_every=1, n_iterations=4000)
    # train('005', 'oxford_flowers', 'resnet50_e512', 'repmet', m=12, d=4, k=3, alpha=2.43)
    # train('006c', 'stanford_dogs', 'resnet50_e1024_fc1024', 'repmet', m=12, d=4, k=3, alpha=1, refresh_clusters=10000)

    """
    NOTES:
    
    1) On flowers data when not normalising the cluster centroids they seem to push away from the unit sphere, generally
        in the positive directions, at least when the emb = 2 (just so i can better vis), not sure if does when higher
        dimension, it might be because in 2 dims the clusters being wrong is too much so pushes away first...
    2) Refreshing the clusters would fix this (clusters moving away too much) to some extent, but it is a balancing act
    3) The learning rate on the centroids for repmet also plays a role, 0.1 or 0.01 is too big and pushes the centroids 
        too far, we find 0.001 to work well
    4) Normalising the centroids helps for faster convergence, and also permits more variation in learning rate
    5) Cluster refresh rate inconclusive at the moment, seems necessary in first epoch or 2 but after not so... but how
        much better is it running kmeans every iteration vs never?
    6) Normalising the centroids and the points means that the dist^2 can only be in range of [0, 4] (4=2^2 where 2 is
        euc dist at opposite ends of unit sphere), this results in post the var_norm and exp that the range is 
        [(far) 0.018, (close) 1], now with summing over the k and k*n_cls clusters we get [0.018*k, k] and 
        [0.018*k*nc, k*nc] for the pre-ln numerator and denominator respectively, this leads to:
            a) the denom will never be smaller than the numer because the numer sum is included (for repmet2, not
                repmet1), therefore the ln() will always be less that 0, meaning the loss will never drop below alpha
            b) in best (perfect) case it will be relu(-ln(k / 0.018*k*(nc-1) + k)+alpha) assuming alpha is 1:
                nc = 102, k=1..K --> relu(2.036) lowest loss
            c) so really the larger that the dist can be, the lower the denom can be but will never reach k meaning
                -ln will always be positive, meaning loss always greater than alpha (for repmet2)
        
        So takeaway, with repmet2 the loss is restricted above alpha, and with cluster norming it is even further
        restricted based on exp(max_dist)*num_classes
    """