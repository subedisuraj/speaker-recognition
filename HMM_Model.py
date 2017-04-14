__author__ = 'pravesh'

from os import listdir, getcwd
from os.path import isfile, join
import pickle
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pylab as pl
import ntpath
from hmm import *

#get the Model directory
model_path = join(getcwd(), "Models")


def get_vectors(filename):
    """
    This file uses pickle to de-serialize the stored model in the Models directory and returns it as result
    raises IO exception
    """
    model = pickle.load(open(join(model_path, filename),"r"))
    return model.codebook


def get_index_hmm(vector, codebook, hmm):
    index_list = []
    i = codebook[0] - vector[0]
    i = i**2

    for items in vector:
        errors = [(sum((codeword-items)**2), index) for index, codeword in enumerate(codebook) if index in hmm.symbol_map]
        index_list.append(min(errors)[1])
    return index_list

def get_index(vector, codebook):
    index_list = []
    i = codebook[0] - vector[0]
    i = i**2

    for items in vector:
        errors = [(sum((codeword-items)**2), index) for index, codeword in enumerate(codebook)]
        index_list.append(min(errors)[1])
    return index_list


def remove_threshold(vector_list, threshold=10.):
    """
    This routine removes the values that are too close together (specified by threshold). For threshold=10, it will
    remove all those values except a single one which lies inside a circle of radius 10
    """
    for vec_x_index in range(len(vector_list)):
        if vector_list[vec_x_index] is None:
            continue
        for vec_y_index in range(vec_x_index, len(vector_list)):
            if vector_list[vec_y_index] is None:
                continue
            if vec_x_index == vec_y_index:
                continue
            error = sum((vector_list[vec_x_index] - vector_list[vec_y_index])**2)
            #print error
            if error <= threshold:
                #TODO: you know what to do
                vector_list[vec_y_index] = None
                pass

    #now condense the list
    #new_vector_list = [i for i in vector_list if i]
    new_vector_list = []
    for i in vector_list:
        if not i is None :
            new_vector_list.append(i)
    return new_vector_list


def adjust():
    """
    There is a difference between other train modules and this train modules. This train module does a bulk train
    i.e. it will take all the model pre-constructed by VQ.py module and generate a long model using them
    """
    allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(join(model_path, obj)) and obj[-2:] == 'VQ']
    vector_list = []

    for files in allfiles:
        vector = get_vectors(files)
        #print sum(vector**2)
        #vector.shape = (vector.shape[0] * vector.shape[1])
        #print sum(vector**2)
        vector_list += [i for i in vector]

    threshold = 10
    new_vector_list = remove_threshold(vector_list, threshold)

    #print len(new_vector_list)
    #At the end, save the total vectors too
    pickle.dump(new_vector_list, open("./Models/codebook.list","w"))


def update(filename):
    """
    This routine will update the codebook using the vectors of id.
    """

    #if the codebook doesn't exist, create from scratch
    if not isfile("./Models/codebook.list"):
        adjust()
        return

    vector = get_vectors(filename)
    codebook = pickle.load(open("./Models/codebook.list","r"))
    codebook = codebook + [np.array(i) for i in vector]
    new_codebook = remove_threshold(codebook)
    pickle.dump(new_codebook, open("./Models/codebook.list","w"))


def train(mfcc, id):
    #now do what must be done
    filename = "./Models/%s.HMM" % id
    codebook = pickle.load(open("./Models/codebook.list","r"))
    quantized_list = get_index(mfcc, codebook)

    #if "%s.HMM" % id in listdir("./Models"):
    if False:
        #update
        newhmm = pickle.load(open(filename, "r"))
        trainedhmm = baum_welch(newhmm, [quantized_list], graph=1, fname="%s" % id)
        pickle.dump(trainedhmm, open(filename, "w"))
    else:
        #create a hmm
        newhmm = HMM(5, V=range(len(codebook)))
        #print newhmm
        trainedhmm = baum_welch(newhmm, [quantized_list],epochs=100, graph=1, fname="%s.png" % id)
        pickle.dump(trainedhmm, open(filename, "w"))


def test(MFCC, verbose=False):
    codebook = pickle.load(open("./Models/codebook.list","r"))
    allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(join(model_path, obj)) and obj[-4:] == '.HMM']
    prob_list = []
    for files in allfiles:
        hmm = pickle.load(open(files,"r"))
        quantized_list = get_index_hmm(MFCC, codebook, hmm)
        prob = forward(hmm, quantized_list, scaling=True)[0]
        prob_list.append((prob, files))

    print ntpath.basename(max(prob_list)[1])[:-4]
    if verbose:
        print
        print "Log likelihoods"
        for logprob, filename in sorted(prob_list, reverse=True):
            print "%-20s has %f" % (ntpath.basename(filename)[:-4], logprob)


def refresh():
    allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(join(model_path, obj)) and obj[-4:] == '.raw']
    for files in allfiles:
        mfcc = pickle.load(open(files, "r"))
        id = ntpath.basename(files)[:-4]
        train(mfcc, id)


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    refresh()
