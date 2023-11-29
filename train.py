# Copyright 2020, Sophos Limited. All rights reserved.
# 
# 'Sophos' and 'Sophos Anti-Virus' are registered trademarks of
# Sophos Limited and Sophos Group. All other product and company
# names mentioned are trademarks or registered trademarks of their
# respective owners.


from dataset import Dataset
from nets import PENetwork, FFNN, BayesWrap
import warnings
import os
import baker
import torch
import torch.nn.functional as F
from torch.utils import data
import sys
from generators import get_generator
from config import device
import config
from logzero import logger
from copy import deepcopy
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import pickle
import json
# import lightgbm as lgb
import argparse
# import wandb

ALPHA = 0.01

def compute_loss(predictions, labels, loss_wts={'malware': 1.0, 'count': 0.1, 'tags': 0.1}):
    """
    Compute losses for a malware feed-forward neural network (optionally with SMART tags 
    and vendor detection count auxiliary losses).

    :param predictions: a dictionary of results from a PENetwork model
    :param labels: a dictionary of labels 
    :param loss_wts: weights to assign to each head of the network (if it exists); defaults to 
        values used in the ALOHA paper (1.0 for malware, 0.1 for count and each tag)
    """
    loss_dict = {'total':0.}
    if 'malware' in labels:
        malware_labels = labels['malware'].float().to(device)
        malware_loss = F.binary_cross_entropy(predictions['malware'].reshape(malware_labels.shape), malware_labels)
        weight = loss_wts['malware'] if 'malware' in loss_wts else 1.0
        loss_dict['malware'] = deepcopy(malware_loss.item())
        loss_dict['total'] += malware_loss * weight
    if 'count' in labels:
        count_labels = labels['count'].float().to(device)
        count_loss = torch.nn.PoissonNLLLoss()(predictions['count'].reshape(count_labels.shape), count_labels)
        weight = loss_wts['count'] if 'count' in loss_wts else 1.0
        loss_dict['count'] = deepcopy(count_loss.item())
        loss_dict['total'] += count_loss * weight
    if 'tags' in labels:
        tag_labels = labels['tags'].float().to(device)
        tags_loss = F.binary_cross_entropy(predictions['tags'], tag_labels)
        weight = loss_wts['tags'] if 'tags' in loss_wts else 1.0
        loss_dict['tags'] = deepcopy(tags_loss.item())
        loss_dict['total'] += tags_loss * weight
    return loss_dict


@baker.command
def train_network(train_db_path=config.db_path,
                  checkpoint_dir=config.checkpoint_dir,
                  max_epochs=10,
                  use_malicious_labels=True,
                  use_count_labels=True,
                  use_tag_labels=True,
                  feature_dimension=2381,
                  random_seed=None, 
                  workers = None,
                  remove_missing_features='scan'):
    """
    Train a feed-forward neural network on EMBER 2.0 features, optionally with additional targets as
    described in the ALOHA paper (https://arxiv.org/abs/1903.05700).  SMART tags based on
    (https://arxiv.org/abs/1905.06262)
    

    :param train_db_path: Path in which the meta.db is stored; defaults to the value specified in `config.py`
    :param checkpoint_dir: Directory in which to save model checkpoints; WARNING -- this will overwrite any existing checkpoints without warning.
    :param max_epochs: How many epochs to train for; defaults to 10
    :param use_malicious_labels: Whether or not to use malware/benignware labels as a target; defaults to True
    :param use_count_labels: Whether or not to use the counts as an additional target; defaults to True
    :param use_tag_labels: Whether or not to use SMART tags as additional targets; defaults to True
    :param feature_dimension: The input dimension of the model; defaults to 2381 (EMBER 2.0 feature size)
    :param random_seed: if provided, seed random number generation with this value (defaults None, no seeding)
    :param workers: How many worker processes should the dataloader use (default None, use multiprocessing.cpu_count())
    :param remove_missing_features: Strategy for removing missing samples, with meta.db entries but no associated features,
        from the data (e.g. feature extraction failures).  
        Must be one of: 'scan', 'none', or path to a missing keys file.  
        Setting to 'scan' (default) will check all entries in the LMDB and remove any keys that are missing -- safe but slow. 
        Setting to 'none' will not perform a check, but may lead to a run failure if any features are missing.  Setting to
        a path will attempt to load a json-serialized list of SHA256 values from the specified file, indicating which
        keys are missing and should be removed from the dataloader.
    """
    workers = workers if workers is None else int(workers)
    os.system('mkdir -p {}'.format(checkpoint_dir))
    if random_seed is not None:
        logger.info(f"Setting random seed to {int(random_seed)}.")
        torch.manual_seed(int(random_seed))
    logger.info('...instantiating network')
    # if args.model == 'ffnn':
    model = FFNN(feature_dimension=feature_dimension, use_malware=True, use_counts=True, use_tags=True, n_tags=len(Dataset.tags)).to(device)
    # else:
    #     model = PENetwork(use_malware=True, use_counts=True, use_tags=True, n_tags=len(Dataset.tags),
                        # feature_dimension=feature_dimension).to(device)
    opt = torch.optim.Adam(model.parameters())
    generator = get_generator(path=train_db_path,
                              mode='train',
                              use_malicious_labels=use_malicious_labels,
                              use_count_labels=use_count_labels,
                              use_tag_labels=use_tag_labels,
                              num_workers = workers,
                              remove_missing_features=remove_missing_features)
    val_generator = get_generator(path = train_db_path,
                                  mode='validation', 
                                  use_malicious_labels=use_malicious_labels,
                                  use_count_labels=use_count_labels,
                                  use_tag_labels=use_tag_labels,
                                  num_workers=workers,
                                  remove_missing_features=remove_missing_features)
    steps_per_epoch = len(generator)
    val_steps_per_epoch = len(val_generator)
    for epoch in range(1, max_epochs + 1):
        loss_histories = defaultdict(list)
        model.train()
        for i, (features, labels) in enumerate(generator):
            opt.zero_grad()
            features = deepcopy(features).to(device)
            out = model(features)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            loss.backward()
            opt.step()
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r Epoch: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch_{}.pt".format(str(epoch))))
        print()
        loss_histories = defaultdict(list)
        model.eval()
        for i, (features, labels) in enumerate(val_generator):
            features = deepcopy(features).to(device)
            with torch.no_grad():
                out = model(features)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r   Val: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, val_steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        print() 
    print('...done')

@baker.command
def train_elbo_network(train_db_path=config.db_path,
                  checkpoint_dir=config.checkpoint_dir,
                  max_epochs=20,
                  use_malicious_labels=True,
                  use_count_labels=True,
                  use_tag_labels=True,
                  feature_dimension=2381,
                  random_seed=None, 
                  workers = None,
                  remove_missing_features='scan'):
    """
    Train a feed-forward neural network on EMBER 2.0 features, optionally with additional targets as
    described in the ALOHA paper (https://arxiv.org/abs/1903.05700).  SMART tags based on
    (https://arxiv.org/abs/1905.06262)
    

    :param train_db_path: Path in which the meta.db is stored; defaults to the value specified in `config.py`
    :param checkpoint_dir: Directory in which to save model checkpoints; WARNING -- this will overwrite any existing checkpoints without warning.
    :param max_epochs: How many epochs to train for; defaults to 10
    :param use_malicious_labels: Whether or not to use malware/benignware labels as a target; defaults to True
    :param use_count_labels: Whether or not to use the counts as an additional target; defaults to True
    :param use_tag_labels: Whether or not to use SMART tags as additional targets; defaults to True
    :param feature_dimension: The input dimension of the model; defaults to 2381 (EMBER 2.0 feature size)
    :param random_seed: if provided, seed random number generation with this value (defaults None, no seeding)
    :param workers: How many worker processes should the dataloader use (default None, use multiprocessing.cpu_count())
    :param remove_missing_features: Strategy for removing missing samples, with meta.db entries but no associated features,
        from the data (e.g. feature extraction failures).  
        Must be one of: 'scan', 'none', or path to a missing keys file.  
        Setting to 'scan' (default) will check all entries in the LMDB and remove any keys that are missing -- safe but slow. 
        Setting to 'none' will not perform a check, but may lead to a run failure if any features are missing.  Setting to
        a path will attempt to load a json-serialized list of SHA256 values from the specified file, indicating which
        keys are missing and should be removed from the dataloader.
    """
    workers = workers if workers is None else int(workers)
    os.system('mkdir -p {}'.format(checkpoint_dir))
    if random_seed is not None:
        logger.info(f"Setting random seed to {int(random_seed)}.")
        torch.manual_seed(int(random_seed))
    logger.info('...instantiating network')
    use_tag_labels = False
    use_count_labels = False
    model = ELBO(feature_dimension=feature_dimension, use_malware=use_malicious_labels, use_counts=use_count_labels, use_tags=use_tag_labels, n_tags=len(Dataset.tags), prior_sigma=2.0).to(device)
    model.load_state_dict(torch.load('/home/harry/SOREL-talos/ckpts/ELBO_var2/epoch_10.pt'))
    opt = torch.optim.Adam(model.parameters())
    generator = get_generator(path=train_db_path,
                              mode='train',
                              use_malicious_labels=use_malicious_labels,
                              use_count_labels=use_count_labels,
                              use_tag_labels=use_tag_labels,
                              num_workers = workers,
                              remove_missing_features=remove_missing_features)
    val_generator = get_generator(path = train_db_path,
                                  mode='validation', 
                                  use_malicious_labels=use_malicious_labels,
                                  use_count_labels=use_count_labels,
                                  use_tag_labels=use_tag_labels,
                                  num_workers=workers,
                                  remove_missing_features=remove_missing_features)
    steps_per_epoch = len(generator)
    val_steps_per_epoch = len(val_generator)
    num_mc = 20
    for epoch in range(11, max_epochs + 1):
        loss_histories = defaultdict(list)
        model.train()
        for i, (features, labels) in enumerate(generator):
            opt.zero_grad()
            features = deepcopy(features).to(device)
            batch_size = features.shape[0]
            output_ = []
            kl_ = []
            for _ in range(num_mc):
                output, kl = model(features)
                output_.append(output)
                kl_.append(kl)
            out_final = {}
            out_final['malware'] = torch.mean(torch.stack([o['malware'] for o in output_]), dim=0)
            # out_final['count'] = torch.mean(torch.stack([o['count'] for o in output_]), dim=0)
            # out = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            loss_dict = compute_loss(out_final, deepcopy(labels))
            loss = loss_dict['total'] + kl / batch_size
            loss.backward()
            opt.step()
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r Epoch: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "epoch_{}.pt".format(str(epoch))))
        print()
        loss_histories = defaultdict(list)
        model.eval()
        for i, (features, labels) in enumerate(val_generator):
            features = deepcopy(features).to(device)
            val_batch_size = features.shape[0]
            with torch.no_grad():
                out, kl = model(features)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total'] + kl / val_batch_size
            for k in loss_dict.keys():
                if k == 'total': loss_histories[k].append(deepcopy(loss_dict[k].detach().cpu().item()))
                else: loss_histories[k].append(loss_dict[k])
            loss_str = " ".join([f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()])
            loss_str += " | "
            loss_str += " ".join([f"{key} mean:{np.mean(value):7.3f}" for key, value in loss_histories.items()])
            sys.stdout.write('\r   Val: {}/{} {}/{} '.format(epoch, max_epochs, i + 1, val_steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels # do our best to avoid weird references that lead to generator errors
        print() 
    print('...done')


@baker.command
def train_ensemble_network(train_db_path=config.db_path,
                        checkpoint_dir=config.checkpoint_dir,
                        max_epochs=config.max_epochs,
                        use_malicious_labels=True,
                        use_count_labels=True,
                        use_tag_labels=True,
                        feature_dimension=2381,
                        random_seed=None,
                        workers=None,
                        remove_missing_features='scan'):
    """
    Train a feed-forward neural network on EMBER 2.0 features, optionally with additional targets as
    described in the ALOHA paper (https://arxiv.org/abs/1903.05700).  SMART tags based on
    (https://arxiv.org/abs/1905.06262)
    
    :param train_db_path: Path in which the meta.db is stored; defaults to the value specified in `config.py`
    :param checkpoint_dir: Directory in which to save model checkpoints; WARNING -- this will overwrite any existing checkpoints without warning.
    :param max_epochs: How many epochs to train for; defaults to 10
    :param use_malicious_labels: Whether or not to use malware/benignware labels as a target; defaults to True
    :param use_count_labels: Whether or not to use the counts as an additional target; defaults to True
    :param use_tag_labels: Whether or not to use SMART tags as additional targets; defaults to True
    :param feature_dimension: The input dimension of the model; defaults to 2381 (EMBER 2.0 feature size)
    :param random_seed: if provided, seed random number generation with this value (defaults None, no seeding)
    :param workers: How many worker processes should the dataloader use (default None, use multiprocessing.cpu_count())
    :param remove_missing_features: Strategy for removing missing samples, with meta.db entries but no associated features,
        from the data (e.g. feature extraction failures).  
        Must be one of: 'scan', None, or path to a missing keys file.  
        Setting to 'scan' (default) will check all entries in the LMDB and remove any keys that are missing -- safe but slow. 
        Setting to 'none' will not perform a check, but may lead to a run failure if any features are missing.  Setting to
        a path will attempt to load a json-serialized list of SHA256 values from the specified file, indicating which
        keys are missing and should be removed from the dataloader.
    """
    workers = workers if workers is None else int(workers)
    os.system('mkdir -p {}'.format(checkpoint_dir))
    if random_seed is not None:
        logger.info(f"Setting random seed to {int(random_seed)}.")
        torch.manual_seed(int(random_seed))
    logger.info('...instantiating network')
    model = PENetwork(use_malware=True,
                      use_counts=True,
                      use_tags=True,
                      n_tags=len(Dataset.tags),
                      feature_dimension=feature_dimension).to(device)

    model = BayesWrap(model, config.num_particles)
    if config.train_mode == 2:
        kwargs = {"return_max": True}  # no entropy required for this
    else:
        kwargs = {"return_max": False}  # no entropy required for this
    print("The process is curr on device {}".format(device))
    opt = torch.optim.Adam(model.parameters())

    start_epoch = 0
    if config.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(
            config.checkpoint_dir), "Error: no checkpoint directory found!"
        # use wandb artifcact
        checkpoint_file = config.checkpoint_file
        checkpoint_file = os.path.join(config.checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint["model"]
        optim_dict = checkpoint["optim"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(state_dict)
        opt.load_state_dict(optim_dict)

    generator = get_generator(path=train_db_path,
                              mode='train',
                              use_malicious_labels=use_malicious_labels,
                              use_count_labels=use_count_labels,
                              use_tag_labels=use_tag_labels,
                              num_workers=workers,
                              remove_missing_features=remove_missing_features)
    val_generator = get_generator(
        path=train_db_path,
        mode='validation',
        use_malicious_labels=use_malicious_labels,
        use_count_labels=use_count_labels,
        use_tag_labels=use_tag_labels,
        num_workers=workers,
        remove_missing_features=remove_missing_features)
    steps_per_epoch = len(generator)
    val_steps_per_epoch = len(val_generator)
    for epoch in range(start_epoch + 1, max_epochs + 1):
        finalLoss = []
        ensLoss = []
        sampledLoss = []
        loss_histories = defaultdict(list)
        log_dict = defaultdict(list)
        model.train()
        for i, (features, labels) in enumerate(generator):
            loss_max = []
            if config.train_mode == 1:
                sampled_model = model.sample_particle()
                sampled_model.train()

            features = deepcopy(features).to(device)
            if config.train_mode == 2:  # max loss
                kwargs = {"return_max": True}  # no entropy required for this
                outs = model(features, **kwargs)  # get multiples outpuut
                for j in range(len(outs)):
                    loss_j_dict = compute_loss(outs[j], deepcopy(labels))
                    loss_j = loss_j_dict['total']
                    loss_max.append(loss_j)
                loss_max = torch.stack(loss_max)
                loss = torch.max(loss_max)
                index = torch.argmax(loss_max)
                loss_dict = compute_loss(outs[index], deepcopy(labels))
            elif config.train_mode == 1:  # sampled loss
                out_sampled = sampled_model(features)
                loss_dict_sampled = compute_loss(out_sampled, deepcopy(labels))
                loss_sampled = loss_dict_sampled['total']
                #  else: # ens loss
                kwargs = {"return_max": False}  # no entropy required for this
                out = model(features, **kwargs)
                loss_dict = compute_loss(out, deepcopy(labels))
                loss = loss_dict['total']

            opt.zero_grad()
            final_loss = loss

            final_loss.backward()

            #  loss.backward()
            #  final_loss.backward()
            # model.update_grads()
            opt.step()

            finalLoss.append(final_loss.item())
            ensLoss.append(loss.item())

            if config.train_mode == 1:
                sampledLoss.append(loss_sampled.item())

            for k in loss_dict.keys():
                if k == 'total':
                    loss_histories[k].append(
                        deepcopy(loss_dict[k].detach().cpu().item()))
                else:
                    loss_histories[k].append(loss_dict[k])

            log_dict = loss_histories

            loss_str = " ".join([
                f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()
            ])
            loss_str += " | "
            loss_str += " ".join([
                f"{key} mean:{np.mean(value):7.3f}"
                for key, value in loss_histories.items()
            ])
            sys.stdout.write('\r Epoch: {}/{} {}/{} '.format(
                epoch, max_epochs, i + 1, steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels  # do our best to avoid weird references that lead to generator errors

        # for key, value in log_dict.items():
            # wandb.log({f"train_{key}loss": np.mean(value)}, step=epoch)

        finalloss = sum(finalLoss) / float(len(finalLoss))
        ensloss = sum(ensLoss) / float(len(ensLoss))

        if config.train_mode == 1:
            sampledloss = sum(sampledLoss) / float(len(sampledLoss))
            # wandb.log({"train_sampled_loss": sampledloss}, step=epoch)

        # wandb.log({"train_final_loss": finalloss}, step=epoch)
        # wandb.log({"train_ens_loss": ensloss}, step=epoch)
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optim": opt.state_dict(),
        }
        torch.save(
            state,
            os.path.join(checkpoint_dir, "epoch_{}.pt".format(str(epoch))))
        print()
        loss_histories = defaultdict(list)
        log_dict = defaultdict(list)
        valLoss = []

        model.eval()

        kwargs = {"return_max": False}  # no entropy required for this
        for i, (features, labels) in enumerate(val_generator):
            features = deepcopy(features).to(device)
            with torch.no_grad():
                out = model(features, **kwargs)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            valLoss.append(loss.item())

            for k in loss_dict.keys():
                if k == 'total':
                    loss_histories[k].append(
                        deepcopy(loss_dict[k].detach().cpu().item()))
                else:
                    loss_histories[k].append(loss_dict[k])
            log_dict = loss_histories
            loss_str = " ".join([
                f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()
            ])
            loss_str += " | "
            loss_str += " ".join([
                f"{key} mean:{np.mean(value):7.3f}"
                for key, value in loss_histories.items()
            ])
            sys.stdout.write('\r   Val: {}/{} {}/{} '.format(
                epoch, max_epochs, i + 1, val_steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels  # do our best to avoid weird references that lead to generator errors
        print()

        # for key, value in log_dict.items():
            # wandb.log({f"val_{key}loss": np.mean(value)}, step=epoch)
        # valloss = sum(valLoss) / float(len(valLoss))
        # wandb.log({"val_loss": valloss}, step=epoch)
    print('...done')

@baker.command
def train_bayes_network(train_db_path=config.db_path,
                        checkpoint_dir=config.checkpoint_dir,
                        max_epochs=config.max_epochs,
                        use_malicious_labels=True,
                        use_count_labels=True,
                        use_tag_labels=True,
                        feature_dimension=2381,
                        random_seed=None,
                        workers=None,
                        remove_missing_features='scan'):
    """
    Train a feed-forward neural network on EMBER 2.0 features, optionally with additional targets as
    described in the ALOHA paper (https://arxiv.org/abs/1903.05700).  SMART tags based on
    (https://arxiv.org/abs/1905.06262)
    
    :param train_db_path: Path in which the meta.db is stored; defaults to the value specified in `config.py`
    :param checkpoint_dir: Directory in which to save model checkpoints; WARNING -- this will overwrite any existing checkpoints without warning.
    :param max_epochs: How many epochs to train for; defaults to 10
    :param use_malicious_labels: Whether or not to use malware/benignware labels as a target; defaults to True
    :param use_count_labels: Whether or not to use the counts as an additional target; defaults to True
    :param use_tag_labels: Whether or not to use SMART tags as additional targets; defaults to True
    :param feature_dimension: The input dimension of the model; defaults to 2381 (EMBER 2.0 feature size)
    :param random_seed: if provided, seed random number generation with this value (defaults None, no seeding)
    :param workers: How many worker processes should the dataloader use (default None, use multiprocessing.cpu_count())
    :param remove_missing_features: Strategy for removing missing samples, with meta.db entries but no associated features,
        from the data (e.g. feature extraction failures).  
        Must be one of: 'scan', None, or path to a missing keys file.  
        Setting to 'scan' (default) will check all entries in the LMDB and remove any keys that are missing -- safe but slow. 
        Setting to 'none' will not perform a check, but may lead to a run failure if any features are missing.  Setting to
        a path will attempt to load a json-serialized list of SHA256 values from the specified file, indicating which
        keys are missing and should be removed from the dataloader.
    """
    workers = workers if workers is None else int(workers)
    os.system('mkdir -p {}'.format(checkpoint_dir))
    if random_seed is not None:
        logger.info(f"Setting random seed to {int(random_seed)}.")
        torch.manual_seed(int(random_seed))
    logger.info('...instantiating network')
    model = PENetwork(use_malware=True,
                      use_counts=True,
                      use_tags=True,
                      n_tags=len(Dataset.tags),
                      feature_dimension=feature_dimension).to(device)

    model = BayesWrap(model, config.num_particles)
    if config.train_mode == 2:
        kwargs = {"return_max": True}  # no entropy required for this
    else:
        kwargs = {"return_max": False}  # no entropy required for this
    print("The process is curr on device {}".format(device))
    opt = torch.optim.Adam(model.parameters())

    start_epoch = 0
    if config.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(
            config.checkpoint_dir), "Error: no checkpoint directory found!"
        # use wandb artifcact
        checkpoint_file = config.checkpoint_file
        checkpoint_file = os.path.join(config.checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint["model"]
        optim_dict = checkpoint["optim"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(state_dict)
        opt.load_state_dict(optim_dict)

    generator = get_generator(path=train_db_path,
                              mode='train',
                              use_malicious_labels=use_malicious_labels,
                              use_count_labels=use_count_labels,
                              use_tag_labels=use_tag_labels,
                              num_workers=workers,
                              remove_missing_features=remove_missing_features)
    val_generator = get_generator(
        path=train_db_path,
        mode='validation',
        use_malicious_labels=use_malicious_labels,
        use_count_labels=use_count_labels,
        use_tag_labels=use_tag_labels,
        num_workers=workers,
        remove_missing_features=remove_missing_features)
    steps_per_epoch = len(generator)
    val_steps_per_epoch = len(val_generator)
    for epoch in range(start_epoch + 1, max_epochs + 1):
        finalLoss = []
        ensLoss = []
        sampledLoss = []
        loss_histories = defaultdict(list)
        log_dict = defaultdict(list)
        model.train()
        for i, (features, labels) in enumerate(generator):
            loss_max = []
            if config.train_mode == 1:
                sampled_model = model.sample_particle()
                sampled_model.train()

            features = deepcopy(features).to(device)
            if config.train_mode == 2:  # max loss
                kwargs = {"return_max": True}  # no entropy required for this
                outs = model(features, **kwargs)  # get multiples outpuut
                for j in range(len(outs)):
                    loss_j_dict = compute_loss(outs[j], deepcopy(labels))
                    loss_j = loss_j_dict['total']
                    loss_max.append(loss_j)
                loss_max = torch.stack(loss_max)
                loss = torch.max(loss_max)
                index = torch.argmax(loss_max)
                loss_dict = compute_loss(outs[index], deepcopy(labels))
            elif config.train_mode == 1:  # sampled loss
                out_sampled = sampled_model(features)
                loss_dict_sampled = compute_loss(out_sampled, deepcopy(labels))
                loss_sampled = loss_dict_sampled['total']
                #  else: # ens loss
                kwargs = {"return_max": False}  # no entropy required for this
                out = model(features, **kwargs)
                loss_dict = compute_loss(out, deepcopy(labels))
                loss = loss_dict['total']

            opt.zero_grad()
            if config.train_mode == 1:
                final_loss = loss + loss_sampled
            else:  # train_mode = 0
                final_loss = loss

            final_loss.backward()

            #  loss.backward()
            #  final_loss.backward()
            model.update_grads()
            opt.step()

            finalLoss.append(final_loss.item())
            ensLoss.append(loss.item())

            if config.train_mode == 1:
                sampledLoss.append(loss_sampled.item())

            for k in loss_dict.keys():
                if k == 'total':
                    loss_histories[k].append(
                        deepcopy(loss_dict[k].detach().cpu().item()))
                else:
                    loss_histories[k].append(loss_dict[k])

            log_dict = loss_histories

            loss_str = " ".join([
                f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()
            ])
            loss_str += " | "
            loss_str += " ".join([
                f"{key} mean:{np.mean(value):7.3f}"
                for key, value in loss_histories.items()
            ])
            sys.stdout.write('\r Epoch: {}/{} {}/{} '.format(
                epoch, max_epochs, i + 1, steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels  # do our best to avoid weird references that lead to generator errors

        # for key, value in log_dict.items():
            # wandb.log({f"train_{key}loss": np.mean(value)}, step=epoch)

        finalloss = sum(finalLoss) / float(len(finalLoss))
        ensloss = sum(ensLoss) / float(len(ensLoss))

        if config.train_mode == 1:
            sampledloss = sum(sampledLoss) / float(len(sampledLoss))
            # wandb.log({"train_sampled_loss": sampledloss}, step=epoch)

        # wandb.log({"train_final_loss": finalloss}, step=epoch)
        # wandb.log({"train_ens_loss": ensloss}, step=epoch)
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optim": opt.state_dict(),
        }
        torch.save(
            state,
            os.path.join(checkpoint_dir, "epoch_{}.pt".format(str(epoch))))
        print()
        loss_histories = defaultdict(list)
        log_dict = defaultdict(list)
        valLoss = []

        model.eval()

        kwargs = {"return_max": False}  # no entropy required for this
        for i, (features, labels) in enumerate(val_generator):
            features = deepcopy(features).to(device)
            with torch.no_grad():
                out = model(features, **kwargs)
            loss_dict = compute_loss(out, deepcopy(labels))
            loss = loss_dict['total']
            valLoss.append(loss.item())

            for k in loss_dict.keys():
                if k == 'total':
                    loss_histories[k].append(
                        deepcopy(loss_dict[k].detach().cpu().item()))
                else:
                    loss_histories[k].append(loss_dict[k])
            log_dict = loss_histories
            loss_str = " ".join([
                f"{key} loss:{value:7.3f}" for key, value in loss_dict.items()
            ])
            loss_str += " | "
            loss_str += " ".join([
                f"{key} mean:{np.mean(value):7.3f}"
                for key, value in loss_histories.items()
            ])
            sys.stdout.write('\r   Val: {}/{} {}/{} '.format(
                epoch, max_epochs, i + 1, val_steps_per_epoch) + loss_str)
            sys.stdout.flush()
            del features, labels  # do our best to avoid weird references that lead to generator errors
        print()

        # for key, value in log_dict.items():
            # wandb.log({f"val_{key}loss": np.mean(value)}, step=epoch)
        # valloss = sum(valLoss) / float(len(valLoss))
        # wandb.log({"val_loss": valloss}, step=epoch)
    print('...done')



if __name__ == '__main__': 
    baker.run()
