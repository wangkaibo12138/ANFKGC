import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step + '_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']
        self.all_ents = list(dataset['ent2id'].keys())

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

        # Calculate node degrees during initialization
        self.calculate_node_degrees()

    def calculate_node_degrees(self):
        """
        计算每个节点的度数并存储在字典中。
        """
        self.node_degrees = {ent: len(rels) for ent, rels in self.e1rel_e2.items()}

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx]
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx]
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel]
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few + self.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # construct support and query negative triples
        support_negative_triples = []
        for triple in support_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            support_negative_triples.append([e1, rel, negative])
        negative_triples = []

        for triple in query_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            negative_triples.append([e1, rel, negative])
        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris:
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
        if len(dis_joint) == 0:
            dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            dis_joint = random.sample(dis_joint, 10)
        for negative in dis_joint:
            negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
            if len(dis_joint) == 0:
                dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative = random.choice(dis_joint)
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        dis_joint = list(set(curr_cand).difference(self.e1rel_e2[e1 + rel] + [e2]))
        for negative in dis_joint:
            negative_triples.append([e1, rel, negative])
        if len(negative_triples) == 0:
            dis_joint = list(set(self.all_ents).difference(self.e1rel_e2[e1 + rel] + [e2]))
            negative_triples = random.sample(dis_joint, 10)
        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    

    def get_well_connected_sources(self, min_neighbors=5):
        """
        获取具有至少 min_neighbors 个邻居的实体列表。
        :param min_neighbors: 至少具有的邻居数。
        :return: 一个包含符合条件的实体的列表。
        """
        well_connected_sources = [node for node, degree in self.node_degrees.items() if degree >= min_neighbors]
        return well_connected_sources

    # 添加的 multi_source_shortest_path_length 方法
    def multi_source_shortest_path_length(self, sources, cutoff=None):
        """
        计算从多个源节点开始的最短路径长度。
        :param sources: 源节点列表
        :param cutoff: 最多允许的跳数
        :return: 一个包含每个节点最短路径的字典
        """
        seen = {}  # level (number of hops) when seen in BFS
        level = 0  # the current level
        nextlevel = set(sources)  # set of nodes to check at next level
        if cutoff is None:
            cutoff = float('inf')

        while nextlevel and cutoff >= level:
            thislevel = nextlevel  # advance to next level
            nextlevel = set()  # and start a new list (fringe)
            for v in thislevel:
              if v not in seen:
                seen[v] = level  # set the level of vertex v
                # 调用 my_edges 函数以获取与节点 v 相连的边
                adj = set(map(lambda x: x[1], self.my_edges(v)))
                nextlevel.update(adj)  # add neighbors of v
            level += 1
        return seen

    def my_edges(self, source):
        """
        返回与给定节点相连的所有边。
        :param source: 源节点
        :return: 相连的边
        """
        edges = self.e1rel_e2.get(source, [])
        for edge in edges:
            yield (source, edge[1])  # 返回源节点和目标节点
