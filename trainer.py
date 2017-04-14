"""
This module is associated with training the models

USAGE:
Simply call python trainer.py filename ID
where, filename is the wav file and ID is the associated identification (Name) of the Speaker

RETURNS
return ERROR_SUCCESS (0) if the trainer returns successfully, else returns -1

"""

__author__ = 'pravesh'

from sys import argv
import sys
import MFCC
import scipy.io.wavfile
import VQ
import os
import pickle
import numpy as np
import HMM_Model

usage = \
"""
USAGE:
Simply call python trainer.py filename ID
where, filename is the wav file and ID is the associated identification (Name) of the Speaker

RETURNS
return ERROR_SUCCESS (0) if the trainer returns successfully, else returns -1"""


def loadWAVfile(filename):
    # Return the sample rate (in samples/sec) and data from a WAV file
    w = scipy.io.wavfile.read(filename)
    x = w[1]    # data is returned as a numpy array with a data-type determined from the file
    fs = w[0]    # Sample rate of wav file as python integer
    return x


def create_file(mfcc, id):
    filename = "./Models/%s.raw" % id
    mfcclist = list(mfcc)
    if os.path.isfile(filename):
        mfcclist += list(pickle.load(open(filename, "r")))
    pickle.dump(np.array(mfcclist), open(filename, "w"))


def train(filename, id):
    rawdata = loadWAVfile(filename)
    mfcc = MFCC.extract(rawdata, show=False)
    model = VQ.Model(id)

    #Train the VQ
    model.train(mfcc)

    #Train the HMM
    create_file(mfcc, id)
    return


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    print os.path.abspath(os.curdir)
    
    sys.stdout = open("stdout.txt", "w")
    sys.stderr = open("stderr.txt", "w")
    print "Training"

    if len(argv) != 3:
        print usage
        exit(0)

    #try:
    train(argv[1], argv[2])
    #except:
    #    print sys.exc_value
    #    exit(-1)
    exit(0)


