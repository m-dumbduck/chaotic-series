import numpy as np
from math import factorial, pi

r = 11
mu = 0.2

def wishart_clustering(samples: np.array, L: int) -> list[np.array]:
    distances = np.linalg.norm(samples[:, np.newaxis, :] - samples[np.newaxis, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    argsorted_distances = np.argsort(distances, axis=1)

    r_nearest_distances = sorted_distances[:, r]
    ordered = np.argsort(r_nearest_distances)

    def double_factorial(n):
        if n <= 1:
            return 1
        return n * double_factorial(n - 2)

    def get_volume_coefficient(L: int):
        if L % 2 == 0:
            return pi ** (L // 2) / factorial((L // 2))
        return 2 ** (L // 2 + 1) * pi ** (L // 2) / double_factorial(L)

    VOLUME_CONST = get_volume_coefficient(L)

    def L_dimensional_volume(arr):
        return arr ** L * VOLUME_CONST

    sz = len(samples)

    p = r / (L_dimensional_volume(r_nearest_distances) * sz)

    w = np.array([-1] * sz)
    completed = np.array([False] * sz)
    clusters = [[] for _ in range(sz)]

    def is_significant(index: int):
        if not clusters[index]:
            return False
        if completed[index]:
            return True
        current_cluster = p[np.array(clusters[index])]
        return np.max(np.abs(current_cluster[:, np.newaxis] - current_cluster[np.newaxis, :])) >= mu

    ind = 1
    for q in ordered:
        nei = argsorted_distances[q, 1:r + 1]
        w_nei = w[nei]
        cs = np.unique(w_nei[w_nei != -1])
        if cs.size == 0:  # not connected to clusters
            w[q] = ind
            clusters[ind].append(q)
            ind += 1
            continue
        cs = cs[~completed[cs]]
        if cs.size == 0:  # all clusters are completed
            w[q] = 0
            clusters[0].append(q)
            continue
        significant = np.array([is_significant(e) for e in cs])
        k = np.sum(significant)
        if k > 1 or cs[0] == 0:
            w[q] = 0
            completed[cs[significant]] = True
            # w[cs[~significant]] = -1
        else:
            for e in cs[1:]:
                clusters[cs[0]] += clusters[e]
            clusters[cs[0]].append(q)
            w[np.isin(w, cs)] = cs[0]
            w[q] = cs[0]

    resulting_clusters = np.unique(w)
    clst = []
    for i in resulting_clusters:
        resulting_indexes = np.array(clusters[i])
        if resulting_indexes.size != 0:
            clst.append(samples[resulting_indexes])
    return clst
