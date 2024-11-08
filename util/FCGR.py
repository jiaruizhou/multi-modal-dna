import math
import collections
from matplotlib import cm
import pylab


def count_kmers(data, sequence, k):
    d = collections.defaultdict(int)
    for i in range(len(data) - (k - 1)):
        d[sequence[i:i + k]] += 1
    for key in list(d.keys()):
        if "N" in key:
            del d[key]
    return d


def probabilities(data, kmer_count, k):
    probabilities = collections.defaultdict(float)
    N = len(data)
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / (N - k + 1)
    return probabilities


def chaos_game_representation(probabilities, k):
    array_size = int(2**k)
    chaos = []
    for i in range(array_size):
        chaos.append([0] * array_size)

    maxx = array_size
    maxy = array_size
    posx = 0
    posy = 0
    for key, value in probabilities.items():
        for char in key:
            if char == "T" or char == "t":
                posx += int(maxx / 2)
            elif char == "C" or char == "c":
                posy += int(maxy / 2)
            elif char == "G" or char == "g":
                posx += int(maxx / 2)
                posy += int(maxy / 2)

            maxx = maxx / 2
            maxy /= 2
        chaos[posy][posx] = value
        maxx = array_size
        maxy = array_size
        posx = 0
        posy = 0
    return chaos


def fcgr(data, seq, k):
    f_k = count_kmers(data.upper(), seq.upper(), k)  # count the number that all subsequency appeared
    f_prob = probabilities(data.upper(), f_k, k)  # normalize the count to probabilities between 0-1
    chaos_k = chaos_game_representation(f_prob, k)  # combine the frequency with the figure in CGR
    return chaos_k
