from collections import defaultdict
import itertools
import math
import numpy as np
from scipy.sparse import coo_matrix, diags

from tqdm import tqdm


def get_window(content_lst, window_size):
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()
        words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)

    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])

    return word_pmi_lst


def get_pmi_edge_lst(content_lst, window_size=20, threshold=0.):

    word_window_freq, word_pair_count, windows_len = get_window(content_lst, window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    print("Total number of edges between words:", len(pmi_edge_lst))

    return pmi_edge_lst

def get_word_doc_freq(content_lst):
    word_doc_list = {}

    for i in tqdm(range(len(content_lst)), desc='Calculate word_doc_freq'):
        doc_words = content_lst[i]
        words = doc_words.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    return word_doc_freq

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
