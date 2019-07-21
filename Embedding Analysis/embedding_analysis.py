# You need to write down your own code here
# Task: Given any head entity name (e.g. Q30) and relation name (e.g. P36), you need to output the top 10 closest tail entity names.
# File entity2vec.vec and relation2vec.vec are 50-dimensional entity and relation embeddings.
# If you use the embeddings learned from Problem 1, you will get extra credits.

import numpy as np

entity_dict = {}
with open('./data/entity2id.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        entity_dict[line[0]] = line[1]

relation_dict = {}
with open('./data/relation2id.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        relation_dict[line[0]] = line[1]

entity_embedding = []
with open('./data/entity2vec.vec', 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        vector = []
        for num in line:
            vector.append(num)
        entity_embedding.append(vector)

relation_embedding = []
with open('./data/relation2vec.vec', 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        vector = []
        for num in line:
            vector.append(num)
        relation_embedding.append(vector)

entity_embedding = np.array(entity_embedding)
relation_embedding = np.array(relation_embedding)


def find_tail():
    head = 'Q30'
    relation = 'P36'
    head_vec = entity_embedding[int(entity_dict[head])]
    relation_vec = relation_embedding[int(relation_dict[relation])]

    cal_tail_vec = []
    for i in range(len(head_vec)):
        cal_tail_vec.append(float(head_vec[i])+float(relation_vec[i]))

    mark_entity = -1
    similarity = np.inf
    for index, element in enumerate(entity_embedding):
        diff = []
        for i in range(len(cal_tail_vec)):
            diff.append(float(cal_tail_vec[i]) - float(element[i]))

        _similarity = np.linalg.norm(np.array(diff))
        if _similarity < similarity:
            mark_entity = index
            similarity = _similarity

    for entity in entity_dict.items():
        if entity[1] == str(mark_entity):
            print(str(entity))
            break


def find_relation():
    head = 'Q30'
    tail = 'Q49'
    head_vec = entity_embedding[int(entity_dict[head])]
    tail_vec = entity_embedding[int(entity_dict[tail])]

    cal_relation_vec = []
    for i in range(len(head_vec)):
        cal_relation_vec.append(float(tail_vec[i])-float(head_vec[i]))

    mark_relation = -1
    similarity = np.inf
    for index, element in enumerate(relation_embedding):
        diff = []
        for i in range(len(cal_relation_vec)):
            diff.append(float(cal_relation_vec[i])-float(element[i]))
        _similarity = np.linalg.norm(np.array(diff))
        if _similarity < similarity:
            mark_relation = index
            similarity = _similarity

    for item in relation_dict.items():
        if item[1] == str(mark_relation):
            print(str(item))
            break


find_tail()
find_relation()
print('Finished!!')

