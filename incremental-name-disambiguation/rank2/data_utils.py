#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json


def load_datas(data_dir):
    with open(os.path.join(data_dir, r'train/train_author.json')) as f:
        train_name_infos = json.load(f)

    with open(os.path.join(data_dir, r'train/train_pub.json')) as f:
        train_paper_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/whole_author_profiles.json')) as f:
        whole_author_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/whole_author_profiles_pub.json')) as f:
        whole_paper_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/cna_valid_unass.json')) as f:
        valid_unass = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/cna_valid_unass_pub.json')) as f:
        valid_paper_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-test/cna_test_unass.json')) as f:
        test_unass = json.load(f)

    with open(os.path.join(data_dir, r'cna-test/cna_test_unass_pub.json')) as f:
        test_paper_infos = json.load(f)

    return train_name_infos, train_paper_infos, whole_author_infos, whole_paper_infos, valid_unass, valid_paper_infos, \
           test_unass, test_paper_infos


def get_author_order_from_valid_papers(unass_datas):
    author_orders = {}
    for line in unass_datas:
        paper, ith = line.strip().split('-')
        if paper not in author_orders:
            author_orders[paper] = set()
        author_orders[paper].add(int(ith))
    return author_orders
