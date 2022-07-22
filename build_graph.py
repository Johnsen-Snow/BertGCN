from collections import defaultdict
from math import log
import random
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils import *


class BuildGraph:
    def __init__(self, dataset):
        self.data_path = f'/home/zhangsen/BertGCN/data/{dataset}.txt'
        self.clean_corpus_path = f'/home/zhangsen/BertGCN/data/corpus/{dataset}.clean.txt'
        self.word_embeddings_dim = 300

        print(f"\n==> 现在的数据集是:{dataset}<==")

        self.content, self.docs = self.get_content()
        self.get_label_list()
        self.vocab = self.get_vocab()


    def get_content(self):
        f = open(self.data_path, 'r')
        lines = f.readlines()
        doc_name_list, doc_test_list, doc_train_list = [], [], []
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
        f.close()

        doc_content_list = []
        f = open(self.clean_corpus_path, 'r')
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
        f.close()

        def get_shuffle_id(doc_list):
            doc_ids = []
            for doc in doc_list:
                doc_id = doc_name_list.index(doc)
                doc_ids.append(doc_id)
            random.shuffle(doc_ids)
            
            return doc_ids
        
        self.train_ids = get_shuffle_id(doc_train_list)
        self.test_ids = get_shuffle_id(doc_test_list)
        self.train_size = len(self.train_ids)
        self.val_size = int(0.1 * self.train_size)
        self.real_train_size = self.train_size - self.val_size
        self.test_size = len(self.test_ids)

        ids = self.train_ids + self.test_ids
        shuffle_doc_words_list = []
        shuffle_doc_name_list = []
        
        for id in ids:
            shuffle_doc_words_list.append(doc_content_list[int(id)])
            shuffle_doc_name_list.append(doc_name_list[int(id)])

        return shuffle_doc_words_list, shuffle_doc_name_list

    def get_label_list(self):
        label_set = set()
        for doc in self.docs:
            label_set.add(doc.split('\t')[2])
        self.label_list = list(label_set)

    def get_vocab(self):
        word_set = set()
        for words in tqdm(self.content, desc='Build vocab'):
            for word in words.split():
                word_set.add(word)
        
        vocab = list(word_set)
        self.vocab_size = len(vocab)
        
        self.word_id_map = {}
        for i in range(self.vocab_size):
            self.word_id_map[vocab[i]] = i

        return vocab

    def get_adj(self):
        row, col, weight = [], [], []
        row, col, weight = self.get_pmi_edge(row, col, weight)
        row, col, weight = self.get_tfidf_edge(row, col, weight)

        node_size = self.train_size + self.vocab_size + self.test_size
        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))

        return adj

    def get_pmi_edge(self, row, col, weight):
        pmi_edge_lst = get_pmi_edge_lst(self.content, window_size=20, threshold=0.0)

        for edge_item in tqdm(pmi_edge_lst, desc='Build edges between words'):
            i = self.word_id_map[edge_item[0]]
            j = self.word_id_map[edge_item[1]]
            row.append(self.train_size + i)
            col.append(self.train_size + j)
            weight.append(edge_item[2])
        
        return row, col, weight


    def get_tfidf_edge(self, row, col, weight):
        doc_word_freq = defaultdict(int)

        for doc_id in range(len(self.content)):
            words = self.content[doc_id]
            for word in words.split():
                word_id = self.word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                doc_word_freq[doc_word_str] += 1

        word_doc_freq = get_word_doc_freq(self.content)

        for i in tqdm(range(len(self.content)), desc='Calculate tfidf between words and docs'):
            words = self.content[i]
            doc_word_set = set()
            for word in words.split():
                if word in doc_word_set:
                    continue
                j = self.word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < self.train_size:
                    row.append(i)
                else:
                    row.append(i + self.vocab_size)
                col.append(self.train_size + j)
                idf = log(1.0 * len(self.content) / word_doc_freq[self.vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)
        
        print("Total number of edges between words and docs:", len(weight))
        
        return row, col, weight

    def get_features(self, data_size, name='all'):
        x = np.zeros((data_size, self.word_embeddings_dim))
        y = list()
        if name == 'train':
            docs = self.docs[:data_size]
        elif name == 'test':
            docs = self.docs[self.train_size:self.train_size+data_size]
        else:
            word_vectors = np.random.uniform(-0.01, 0.01, (self.vocab_size, self.word_embeddings_dim))
            x = np.vstack((x, word_vectors))
            docs = self.docs[:data_size]
        
        for doc in docs:
            label = doc.split('\t')[2]
            one_hot = [0 for _ in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)

        y = np.array(y)

        if name == 'all':
            one_hot = [0 for _ in range(len(self.label_list))]
            one_hots = [one_hot for _ in range(self.vocab_size)]
            one_hots = np.array(one_hots)
            y = np.vstack((y, one_hots))

        return x, y

    def load_corpus(self):
        x, y = self.get_features(self.real_train_size, name='train')
        tx, ty = self.get_features(self.test_size, name='test')
        allx, ally = self.get_features(self.train_size, name='all')
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

        features = np.vstack((allx, tx))
        labels = np.vstack((ally, ty))
        print(len(labels[2]))

        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + self.val_size)
        idx_test = range(allx.shape[0], allx.shape[0] + self.test_size)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        adj = self.get_adj()
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
