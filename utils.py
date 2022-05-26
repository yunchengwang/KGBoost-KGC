import os
import random


def load_dataset(data_dir):
    entity2id = dict()
    relation2id = dict()
    id2entity = dict()
    id2relation = dict()

    with open(os.path.join(data_dir, 'entities.dict'), 'r+') as f:
        for line in iter(f):
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_dir, 'relations.dict'), 'r+') as f:
        for line in iter(f):
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nrelation = len(relation2id)

    with open(os.path.join(data_dir, 'entity2text.txt'), 'r+') as f:
        for line in iter(f):
            ent, text = line.split('\t')
            if ent in entity2id:
                id2entity[entity2id[ent]] = text.strip()

    with open(os.path.join(data_dir, 'relation2text.txt'), 'r+') as f:
        for line in iter(f):
            rel, text = line.split('\t')
            if rel in relation2id:
                id2relation[relation2id[rel]] = rel
                id2relation[relation2id[rel] + nrelation] = '(inverse)' + rel

    data = {"train": [], "valid": [], "test": []}

    with open(os.path.join(data_dir, "train.txt"), 'r+') as f:
        for line in iter(f):
            head, relation, tail = line.split()
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            data['train'].append((head, relation, tail))
            data['train'].append((tail, relation + nrelation, head))

    data['train'] = list(set(data['train']))

    with open(os.path.join(data_dir, "valid.txt"), 'r+') as f:
        for line in iter(f):
            head, relation, tail = line.split()
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            data['valid'].append((head, relation, tail))
            data['valid'].append((tail, relation + nrelation, head))

    with open(os.path.join(data_dir, "test.txt"), 'r+') as f:
        for line in iter(f):
            head, relation, tail = line.split()
            head, tail = entity2id[head], entity2id[tail]
            relation = relation2id[relation]
            data['test'].append((head, relation, tail))
            data['test'].append((tail, relation + nrelation, head))

    data['complete'] = data['train'] + data['valid'] + data['test']

    return len(entity2id), len(relation2id), data, id2entity, id2relation


def calc_lcw_index(train_triples_by_relation, k=5):
    error = dict()
    n_triples = dict()
    train_triples = []
    for r in train_triples_by_relation:
        error[r] = 0
        n_triples[r] = len(train_triples_by_relation[r])
        for h, t in train_triples_by_relation[r]:
            train_triples.append((h, r, t))

    random.shuffle(train_triples)

    for i in range(k):
        valid, train = train_triples[i * len(train_triples) // k:(i + 1) * len(train_triples) // k], \
                       train_triples[: i * len(train_triples) // k] + train_triples[(i + 1) * len(train_triples) // k:]
        relation_range = dict()
        for h, r, t in train:
            if r not in relation_range:
                relation_range[r] = set()
            relation_range[r].add(t)

        for h, r, t in valid:
            if r in set(relation_range):
                if t not in relation_range[r]:
                    error[r] += 1

    lcw_index = dict()

    for r in train_triples_by_relation:
        lcw_index[r] = 1.0 - error[r] / n_triples[r]

    return lcw_index
