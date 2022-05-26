import numpy as np
from utils import load_dataset, calc_lcw_index
import xgboost as xgb
import os
import logging
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='XGBoost classifier for Knowledge Graph Completion',
        usage='KGBoost.py [<args>] [-h | --help]'
    )

    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='FB15k-237', type=str)
    parser.add_argument('--emb_dir', default='pretrained', type=str)
    parser.add_argument('-o', '--output_dir', default='output', type=str)
    parser.add_argument('--pretrained_emb', default='TransE', type=str)
    parser.add_argument('--max_depth', default=5, type=int)
    parser.add_argument('--negative_size', default=64, type=int)
    parser.add_argument('--n_estimators', default=1000, type=int)
    parser.add_argument('--sampling_1', default='rcwc')
    parser.add_argument('--sampling_2', default='none')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--lcwa', action='store_true')
    parser.add_argument('--lcw_threshold', default=0.9, type=float)

    return parser.parse_args(args)


def set_logger(args):
    log_path = "{}/{}_{}".format(args.output_dir, args.pretrained_emb, args.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, '{}_{}.log'.format(args.sampling_1, args.sampling_2))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(xgb.__version__)
    logging.info('Augment: {}'.format(args.augment))
    logging.info('Initial negative sampling: {}'.format(args.sampling_1))
    logging.info('Boosting negative sampling: {}'.format(args.sampling_2))
    logging.info('# of negative samples: {}'.format(args.negative_size))
    logging.info('max depth: {}'.format(args.max_depth))
    logging.info('# of estimators: {}'.format(args.n_estimators))
    logging.info('LCWA-based link prediction: {}'.format(args.lcwa))
    if args.lcwa:
        logging.info('lcw threshold: {}'.format(args.lcw_threshold))


def main(args):
    logging.info('Using GPU: {}'.format(args.gpu))

    n_entity, n_relation, data, id2entity, id2relation = load_dataset(os.path.join(args.data_dir, args.dataset))

    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('# of entities: {}'.format(n_entity))
    logging.info('# of relations: {}'.format(n_relation))

    entity_pair_dict = dict()

    for h, r, t in data['train']:
        if (h, t) not in entity_pair_dict:
            entity_pair_dict[(h, t)] = len(entity_pair_dict)

    triple_by_relation = {key: {} for key in data.keys()}

    for split in data:
        for h, r, t in data[split]:
            if r not in triple_by_relation[split]:
                triple_by_relation[split][r] = []
            triple_by_relation[split][r].append((h, t))

    logging.info('Triples grouped by relations...')

    relation_cooccurrence = np.zeros((n_relation * 2, len(entity_pair_dict)))
    for r in triple_by_relation['train']:
        for h, t in triple_by_relation['train'][r]:
            relation_cooccurrence[r][entity_pair_dict[(h, t)]] = 1

    relation_cooccurrence = np.matmul(relation_cooccurrence, relation_cooccurrence.T)
    relation_inference = np.zeros(relation_cooccurrence.shape)

    for i in range(relation_inference.shape[0]):
        for j in range(relation_inference.shape[1]):
            relation_inference[i, j] = \
                1.0 * relation_cooccurrence[i, j] / relation_cooccurrence[j, j]

    logging.info('Relation inference matrix constructed...')

    # Augmenting positive triples
    if args.augment:
        for r1 in range(relation_cooccurrence.shape[0]):
            sub_threshold = 0.8
            for r2 in range(relation_cooccurrence.shape[1]):
                if r1 == r2:
                    continue
                if relation_inference[r1, r2] >= sub_threshold:
                    logging.info('{} borrows triples from {}'.format(id2relation[r1], id2relation[r2]))
                    triple_by_relation['train'][r1] += triple_by_relation['train'][r2]

            triple_by_relation['train'][r1] = list(set(triple_by_relation['train'][r1]))

        logging.info('Positive triples augmented...')

    relation_range = dict()
    for r in triple_by_relation['train']:
        for h, t in triple_by_relation['train'][r]:
            if r not in relation_range:
                relation_range[r] = []
            relation_range[r].append(t)
            relation_range[r] = list(set(relation_range[r]))

    head_dict = dict()
    tail_dict = dict()

    for r in relation_range:
        head_dict[r] = {relation_range[(r + n_relation) % (2 * n_relation)][i]: i
                        for i in range(len(relation_range[(r + n_relation) % (2 * n_relation)]))}
        tail_dict[r] = {relation_range[r][i]: i for i in range(len(relation_range[r]))}

    lcw_index = calc_lcw_index(triple_by_relation['train'])

    if args.lcwa:
        excluded_entity = dict()
        for r in range(n_relation * 2):
            excluded_entity[r] = []
            if r in relation_range and lcw_index[r] >= args.lcw_threshold:
                # logging.info('{} is range-constrained: {}'.format(id2relation[r], lcw_index[r]))
                for i in range(n_entity):
                    if i not in set(relation_range[r]):
                        excluded_entity[r].append(i)

    record_metrics = dict()

    for r in triple_by_relation['test']:
        record_metrics[r] = {'MR': 0.0, 'MRR': 0.0, 'H@1': 0.0, 'H@3': 0.0, 'H@10': 0.0, 'n_test': 0}
        record_metrics[r]['n_test'] = len(triple_by_relation['test'][r])

    negative_sampling_space = dict()

    if 'rcwc' in [args.sampling_1, args.sampling_2]:
        tail_cooccurrence = dict()

        for r in triple_by_relation['train']:
            tail_cooccurrence[r] = np.zeros((len(tail_dict[r]), len(head_dict[r])))
            for h, t in triple_by_relation['train'][r]:
                tail_cooccurrence[r][tail_dict[r][t]][head_dict[r][h]] = 1
            tail_cooccurrence[r] = np.matmul(tail_cooccurrence[r], tail_cooccurrence[r].T)

        logging.info('Tail co-occurrence established...')

        # Sample from low co-occurrence attributes
        for r in tail_cooccurrence:
            for i in range(tail_cooccurrence[r].shape[0]):
                negative_sampling_space[(-1, r, relation_range[r][i])] = []
                threshold = np.percentile(tail_cooccurrence[r][i], 90)
                for j in range(tail_cooccurrence[r].shape[1]):
                    if tail_cooccurrence[r][i, j] <= threshold:
                        negative_sampling_space[(-1, r, relation_range[r][i])].append(relation_range[r][j])

    filter = {'train': dict(), 'complete': dict()}

    for split in filter:
        for r in triple_by_relation[split]:
            for h, t in triple_by_relation[split][r]:
                if (h, r) not in filter[split]:
                    filter[split][(h, r)] = set()
                filter[split][(h, r)].add(t)

    entity_embedding = np.load('{}/{}_{}.npy'.format(args.emb_dir,
                                                     args.pretrained_emb,
                                                     args.dataset))

    logging.info('Pretrained embedding {} loaded'.format(args.pretrained_emb))
    if args.pretrained_emb == 'RotatE':
        logging.info('Pretrained embedding dimension: {}'.format(entity_embedding.shape[1] // 2))
    else:
        logging.info('Pretrained embedding dimension: {}'.format(entity_embedding.shape[1]))

    for r in triple_by_relation['complete']:
        logging.info("=== Current relation: {} ===".format(id2relation[r]))

        #  Best configurations for WordNet subsets
        if id2relation[r] in ['_hypernym', '(inverse)_hypernym', '(inverse)_member_meronym',
                              '_member_meronym', '(inverse)_derivationally_related_form']:
            max_depth = 10
            n_estimators = 1500
        elif id2relation[r] in ['_derivationally_related_form', '_has_part', '(inverse)_has_part']:
            max_depth = 5
            n_estimators = 1500
        else:
            max_depth = args.max_depth
            n_estimators = args.n_estimators

        clf = xgb.XGBClassifier(objective='binary:logistic',
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                use_label_encoder=False,
                                tree_method='gpu_hist',
                                gpu_id=args.gpu)

        if r in triple_by_relation['train']:
            X_train = []
            X_train_negative = []
            y_train = []
            X_train_boost = []
            y_train_boost = []
            for h, t in triple_by_relation['train'][r]:
                X_train.append(np.concatenate((entity_embedding[h], entity_embedding[t])))
                y_train.append(1)

                if args.sampling_2 != 'none':
                    X_train_boost.append(np.concatenate((entity_embedding[h], entity_embedding[t])))
                    y_train_boost.append(1)

                    if args.sampling_2 != 'adv':
                        if args.sampling_2 == 'rcwc':
                            if (-1, r, t) in negative_sampling_space:
                                negative_sample = np.random.randint(len(negative_sampling_space[(-1, r, t)]),
                                                                    size=args.negative_size)
                                for i in range(args.negative_size):
                                    negative_sample[i] = negative_sampling_space[(-1, r, t)][negative_sample[i]]
                            else:
                                negative_sample = np.random.randint(n_entity, size=args.negative_size)
                        else:
                            raise ValueError('Negative sampling not supported')

                        mask = np.in1d(
                            negative_sample,
                            list(filter['train'][(h, r)]),
                            assume_unique=True,
                            invert=True
                        )
                        negative_sample = negative_sample[mask]

                        if len(negative_sample) == 0:
                            negative_sample = np.random.randint(n_entity, size=args.negative_size)
                            mask = np.in1d(
                                negative_sample,
                                list(filter['train'][(h, r)]),
                                assume_unique=True,
                                invert=True
                            )
                            negative_sample = negative_sample[mask]

                        for i in range(len(negative_sample)):
                            X_train_boost.append(
                                np.concatenate((entity_embedding[h], entity_embedding[negative_sample[i]])))
                            y_train_boost.append(0)

                if args.sampling_1 == 'naive':
                    negative_sample = np.random.randint(n_entity, size=args.negative_size)

                elif args.sampling_1 == 'rcwc':
                    if (-1, r, t) in negative_sampling_space:
                        negative_sample = np.random.randint(len(negative_sampling_space[(-1, r, t)]),
                                                            size=args.negative_size)
                        for i in range(args.negative_size):
                            negative_sample[i] = negative_sampling_space[(-1, r, t)][negative_sample[i]]
                    else:
                        negative_sample = np.random.randint(n_entity, size=args.negative_size)
                else:
                    raise ValueError('Initial negative sampling {} not supported'.format(args.sampling_1))

                mask = np.in1d(
                    negative_sample,
                    list(filter['train'][(h, r)]),
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]

                if len(negative_sample) == 0:
                    negative_sample = np.random.randint(n_entity, size=args.negative_size)
                    mask = np.in1d(
                        negative_sample,
                        list(filter['train'][(h, r)]),
                        assume_unique=True,
                        invert=True
                    )
                    negative_sample = negative_sample[mask]

                for i in range(len(negative_sample)):
                    X_train.append(np.concatenate((entity_embedding[h], entity_embedding[negative_sample[i]])))
                    X_train_negative.append(np.concatenate((entity_embedding[h], entity_embedding[negative_sample[i]])))
                    y_train.append(0)

            X_train = np.array(X_train)
            X_train_negative = np.array(X_train_negative)
            y_train = np.array(y_train)

            logging.info("=== Dataset Constructed ===")
            logging.info("=== Start Training      ===")
            logging.info("# of training samples: {}".format(y_train.shape[0]))

            clf.fit(X_train, y_train)

            if args.sampling_2 != 'none':
                if args.sampling_2 == 'adv':
                    for i in range(len(X_train_negative)):
                        X_train_boost.append(X_train_negative[i])
                        y_train_boost.append(0)

                    previous_score = clf.predict_proba(X_train_negative)[:, 1]
                    adversarial_negative_sample = X_train_negative[
                        np.argsort(-previous_score)[:len(X_train_negative) // 2]]
                    for i in range(len(adversarial_negative_sample)):
                        X_train_boost.append(adversarial_negative_sample[i])
                        y_train_boost.append(0)
                X_train_boost = np.array(X_train_boost)
                y_train_boost = np.array(y_train_boost)
                clf.fit(X_train_boost, y_train_boost, xgb_model=clf.get_booster())

            logging.info("=== Classifier Trained  ===")
            logging.info("=== Start Evaluation    ===")

        mr = 0
        mrr = 0
        hits_1 = 0
        hits_3 = 0
        hits_10 = 0

        if r not in triple_by_relation['test']:
            logging.info('Relation {} not in testing set'.format(id2relation[r]))
            continue
        for h, t in triple_by_relation['test'][r]:
            all_entity = np.arange(n_entity)
            all_entity_embedding = entity_embedding[all_entity]
            head_embedding = entity_embedding[h]
            head_embedding = np.tile(head_embedding, (n_entity, 1))
            X_test = np.concatenate((head_embedding, all_entity_embedding), axis=1)

            pred = clf.predict_proba(X_test)
            score = pred[:, 1]

            if args.lcwa:
                exclusion = list(set(excluded_entity[r] + list(filter['complete'][(h, r)])))
            else:
                exclusion = list(filter['complete'][(h, r)])
            exclusion.remove(t)
            exclusion = np.array(exclusion, dtype=int)
            score[exclusion] = 0
            argsort = np.argsort(-score)

            ranking = (argsort == t).nonzero()[0][0] + 1

            mr += ranking
            mrr += 1.0 / ranking
            hits_1 += int(ranking <= 1)
            hits_3 += int(ranking <= 3)
            hits_10 += int(ranking <= 10)

        record_metrics[r]['MR'] = mr / record_metrics[r]['n_test']
        record_metrics[r]['MRR'] = mrr / record_metrics[r]['n_test']
        record_metrics[r]['H@1'] = hits_1 / record_metrics[r]['n_test']
        record_metrics[r]['H@3'] = hits_3 / record_metrics[r]['n_test']
        record_metrics[r]['H@10'] = hits_10 / record_metrics[r]['n_test']

        logging.info('n_test: {}'.format(record_metrics[r]['n_test']))
        logging.info('MR: {}'.format(record_metrics[r]['MR']))
        logging.info('MRR: {}'.format(record_metrics[r]['MRR']))
        logging.info('H@1: {}'.format(record_metrics[r]['H@1']))
        logging.info('H@3: {}'.format(record_metrics[r]['H@3']))
        logging.info('H@10: {}\n'.format(record_metrics[r]['H@10']))

    total_mr = 0
    total_mrr = 0
    total_hits_1 = 0
    total_hits_3 = 0
    total_hits_10 = 0
    total_ntest = 0
    for r in record_metrics:
        total_mr += record_metrics[r]['n_test'] * record_metrics[r]['MR']
        total_mrr += record_metrics[r]['n_test'] * record_metrics[r]['MRR']
        total_hits_1 += record_metrics[r]['n_test'] * record_metrics[r]['H@1']
        total_hits_3 += record_metrics[r]['n_test'] * record_metrics[r]['H@3']
        total_hits_10 += record_metrics[r]['n_test'] * record_metrics[r]['H@10']
        total_ntest += record_metrics[r]['n_test']

    logging.info('Total number of the testing samples: {}'.format(total_ntest))

    total_mr = total_mr / total_ntest
    total_mrr = total_mrr / total_ntest
    total_hits_1 = total_hits_1 / total_ntest
    total_hits_3 = total_hits_3 / total_ntest
    total_hits_10 = total_hits_10 / total_ntest

    logging.info('MR: {}'.format(total_mr))
    logging.info('MRR: {}'.format(total_mrr))
    logging.info('H@1: {}'.format(total_hits_1))
    logging.info('H@3: {}'.format(total_hits_3))
    logging.info('H@10: {}'.format(total_hits_10))

    text_path = "{}/{}_{}".format(args.output_dir, args.pretrained_emb, args.dataset)
    text_file = os.path.join(text_path, '{}_{}.txt'.format(args.sampling_1, args.sampling_2))

    with open(text_file, 'w') as f:
        for r in record_metrics:
            f.write('Relation: {}\n'.format(id2relation[r]))
            f.write('n_test: {}\n'.format(record_metrics[r]['n_test']))
            f.write('MR: {}\n'.format(record_metrics[r]['MR']))
            f.write('MRR: {}\n'.format(record_metrics[r]['MRR']))
            f.write('H@1: {}\n'.format(record_metrics[r]['H@1']))
            f.write('H@3: {}\n'.format(record_metrics[r]['H@3']))
            f.write('H@10: {}\n\n'.format(record_metrics[r]['H@10']))


if __name__ == '__main__':
    args = parse_args()
    set_logger(args)
    main(args)
