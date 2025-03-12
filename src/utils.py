#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import heapq
import torch
import numpy as np
import random
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import (
    cifar_iid,
    cifar_noniid,
    cifar_extr_noniid,
    miniimagenet_extr_noniid,
    mnist_extr_noniid,
)


def get_mal_dataset(dataset, num_mal, num_classes, label_tampering):
    X_list = np.random.choice(len(dataset), num_mal)
    print(X_list)
    Y_true = []
    for i in X_list:
        _, Y = dataset[i]
        Y_true.append(Y)
    Y_mal = []

    if label_tampering == "zero":
        Y_mal = [0 for _ in range(num_mal)]
    elif label_tampering == "random":
        Y_mal = [random.randint(0, num_classes - 1) for _ in range(num_mal)]
    elif label_tampering == "reverse":
        Y_mal = [num_classes - 1 - y for y in Y_true]
    elif label_tampering == "none":
        Y_mal = Y_true

    return X_list, Y_mal, Y_true


def get_dataset(args):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    def unbalanced_classes(test_val_index, targets):
        # targets = [targets[i] for i in test_val_index]
        under_represented_classes = random.sample(range(10), 3)
        class_limit = 500
        class_dist = [0 for _ in range(10)]
        new_test_val_index = []
        for i, idx in enumerate(test_val_index):
            if targets[i] in under_represented_classes:
                if class_dist[targets[i]] < class_limit:
                    new_test_val_index.append(idx)
                    class_dist[targets[i]] += 1
            else:
                new_test_val_index.append(idx)

        return new_test_val_index

    def create_subset(dataset, indices: list[int]):
        subset_data = dataset.data[indices]
        subset_targets = dataset.targets[indices]

        dataset_subset = type(dataset)(
            root=dataset.root,
            train=True,
            transform=dataset.transform,
            download=False,
        )
        dataset_subset.data = subset_data
        dataset_subset.targets = subset_targets

        return dataset_subset

    if args.dataset == "cifar":
        data_dir = "../data/cifar/"
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == "mnist" or "fmnist":
        if args.dataset == "mnist":
            data_dir = "../data/mnist/"
        else:
            data_dir = "../data/fashion_mnist/"

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if args.dataset == "mnist":
            data_full = datasets.MNIST(
                data_dir, train=True, download=True, transform=apply_transform
            )
        else:
            train_dataset = datasets.FashionMNIST(
                data_dir, train=True, download=True, transform=apply_transform
            )
            test_dataset = datasets.FashionMNIST(
                data_dir, train=False, download=True, transform=apply_transform
            )

        train_idx, test_idx = train_test_split(
            range(len(data_full)), test_size=0.2, stratify=data_full.targets
        )
        if isinstance(data_full.targets, list):
            data_full.targets = torch.tensor(data_full.targets)  # type: ignore
        test_idx = unbalanced_classes(test_idx, data_full.targets[test_idx])

        train_dataset = create_subset(data_full, train_idx)
        test_dataset = create_subset(data_full, test_idx)
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)

        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_ns(w, ns):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] * ns[0]
        for i in range(1, len(w)):
            w_avg[key] += ns[i] * w[i][key]
        w_avg[key] = torch.div(w_avg[key], sum(ns))
    return w_avg


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return
