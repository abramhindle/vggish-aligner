#!/bin/env python3
import argparse
import random
import numpy as np
import numpy
from pippi import dsp
import scipy
import functools
import itertools
import ctypes
import scipy.spatial
import scipy
sr=48000


@functools.lru_cache(maxsize=30)
def get_whole_file_samples(filename):
    return dsp.read(filename)


@functools.lru_cache(maxsize=256)
def get_samples(filename_tuple,sr=48000):
    samples = get_whole_file_samples( filename_tuple[0] )
    i = sr*filename_tuple[1]
    return samples[i:(i+sr)]

def calculate_similarity(ttfeatures, dbmat, sim='cosine'):
    return scipy.spatial.distance.cdist(ttfeatures, dbmat, metric=sim)
    

def fit_to_input(ttres):
    ttfilt = np.nan_to_num(ttres,nan=np.Inf,copy=False)
    ttfit = np.argmin(ttfilt,axis=1)
    return ttfit

def most_similar(row, dbmat, sim='cosine'):
    return fit_to_input(calculate_similarity( row, dbmat, sim=sim))

def most_similar_all(rows, dbmat, sim='cosine'):
    similar = np.zeros(rows.shape[0])
    for i in range(rows.shape[0]):
        row = rows[i:i+1,:]
        similar[i] = most_similar(row,dbmat,sim=sim)[0]
    return similar

def render(ttfeatures, ttfit, db,freq=1.0, step=1.0):
    out = dsp.buffer(length=step*ttfeatures.shape[0]/sr, samplerate=sr)
    skip = sr//int((1/freq)*step)
    steps = ttfit.shape[0] // skip
    for i in range(steps):
        j = i * skip
        index = int(ttfit[j])
        s = get_samples(db["filenames"][index])
        t = j * step / sr
        # print(t,index)
        out.dub(s,t)
    return out


def new_render(ttfeatures, ttfit, db,step=1.0):
    length = step*ttfeatures.shape[0]
    print(f"length in seconds: {length}")
    out = dsp.buffer(length=step*ttfeatures.shape[0], samplerate=sr)
    for i in range(ttfit.shape[0]):
        index = int(ttfit[i])
        s = get_samples(db["filenames"][index])
        t = i * step
        out.dub(s,t)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='Align a long sound to a DB of sounds')
    parser.add_argument('-i', help='Input wave csv')
    parser.add_argument('-out',default='output.wav', help='the rendered output')
    parser.add_argument('-csv',default="out.csv",help="input csv file")
    parser.add_argument('-s', default=1.0,help='frequency per second of sounds')
    parser.add_argument('-sim', default='cosine',help='similarity [cosine, canberra, dice, yule, hamming, correlation, euclidean, sqeuclidean]')
    args = parser.parse_args()
    return args

def read_db(csvfile):
    db = {"filenames":None,"mat":None}
    rows = []
    filenames = []
    with open(csvfile) as fd:
        for line in fd.readlines():
            out = line.strip().split(',')
            filename, index = out[0],out[1]
            index = int(index)
            rest = out[2:]
            rest = [int(y) for y in rest]
            filenames.append((filename,index))
            rows.append(rest)
    db["mat"] = numpy.array(rows)
    print(csvfile,db["mat"].shape)
    db["filenames"] = filenames
    return db

if __name__ == "__main__":
    args = parse_args()
    filename = args.i
    print("Loading %s" % filename)
    input_db = read_db( filename )
    print("Loading %s" % args.csv)
    db = read_db(args.csv)
    print("Calculating Similarity")
    print(input_db["mat"])
    print(db["mat"])

    closest = most_similar_all(input_db["mat"],db["mat"])
    # kdtree = scipy.spatial.cKDTree(db["mat"])
    # closest = kdtree.query(input_db["mat"],k=1)[1]
    print(closest) 
    #ttres = calculate_similarity(input_db["mat"], db["mat"], sim=args.sim)
    #print("Filtering")
    #ttfit = fit_to_input(ttres)
    #print(ttfit.shape)
    #print("Rendering")
    freq = 1.0
    out = new_render(input_db["mat"], closest, db, freq)
    out.write(args.out)
