"""
This module is associated with the Vector Quantization approach for training the data. Before jumping into
HMM, I thought "hey, why not try VQ first?" and thus the module was born.

In addition to being a starter, this module actually shows the abstraction that we will be doing in all other
feature recognition modules. These abstractions are as follows

Model class

Each of these will be documented in their own method.
"""

from os import listdir, getcwd
from os.path import isfile, join
import os
import pickle
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pylab as pl
import HMM_Model

os.chdir(os.path.dirname(__file__))

#get the Model directory
model_path = join(getcwd(), "Models")

#number of codewords in a codebook
M = 32


class Model:
    """
    This class provides the base for tasks like training and testing.
    """
    def __init__(self, id):
        """
        The initializer for this class. id is the speaker id of this model. This class is only supposed to be used
        explicitly under the training phase. For the testing phase, a helper function should be called, which unpickles
        each module and calculates the score.
        """
        self.id = id
        self.filename = join(model_path, self.id+'.VQ')

    def absolute_train(self, data):
        self.codebook = Model.get_codebook(data)
        #save the data so that we may use it some other time (e.g. visualization)
        self.data = data

    @staticmethod
    def get_codebook( data):
        """
        Train from scratch
        Returns the codebook
        """
        num_vectors, num_dimensions = data.shape

        #at first, there is only one codeword, the centroid of everything
        first = sum(data)/float(num_vectors)

        codebook = np.zeros((1, num_dimensions))
        codebook[0] = first.copy()
        num_codeword = 1

        #initialize the random number generator
        random.seed()

        while True:
            #split the codeword into two.

            new_codebook = np.zeros((num_codeword*2, num_dimensions))

            for index, codewords in enumerate(codebook):
                #generate a delta to randomly split the centroid

                #delta_plus = 1+random(), delta_minus = 1-random()
                delta_plus = np.array([1+random.random() for i in range(num_dimensions)], dtype=np.float64)
                delta_minus = 2 - delta_plus

                new_codebook[2*index] = codewords * delta_plus
                new_codebook[2*index+1] = codewords * delta_minus

            #This is used for reference purposes, to check the deviation
            new_codebook_backup = new_codebook.copy()

            #Repeat until a certain threshold is encountered
            while True:
                #keep a counter of the number of vectors in each partition
                num_vectors_in_partition = np.zeros(num_codeword*2) + 1
                partition = np.zeros(new_codebook.shape)

                #after the new codebook is initialized, we now iterate over all the samples again and
                #reposition the centroids (Nearest Neighbour search)
                for datum in data:
                    #find the nearest codeword

                    #distance is the list of actual distance of vector datum and the codewords. We now have to find the
                    #nearest codeword
                    distance = map(lambda x: ( sum((datum-x[1])**2), x[0]), enumerate(new_codebook))

                    #now distance is of the form [(actual distance, codeword index)....]
                    min_distance = min(distance)
                    #hence, the nearest code word is min_distance[1]

                    #now update the nearest codeword
                    partition[min_distance[1]] += datum

                    #update the number of vectors in the partition of corresponding minimum codeword
                    num_vectors_in_partition[min_distance[1]] += 1

                new_codebook = np.array(map(lambda x: x[1]/num_vectors_in_partition[x[0]], enumerate(partition)))

                threshold = (new_codebook_backup - new_codebook)**2
                threshold = np.array(map(lambda x: sum(x), threshold))
                new_codebook_backup = new_codebook.copy()
                if all(threshold < 0.01):
                    #The codewords have stabilized
                    break


            num_codeword *= 2
            if num_codeword > M/2:
                break

            codebook = new_codebook.copy()

        #save the training data
        return codebook

    def visualize(self):
        """
        This method uses PCA to reduce the dimensions of the trained data as well as the VQ codebook
        and then displays them in a 2D graph
        """
        pca = PCA(n_components=2,whiten=True).fit(self.data)
        data_pca = pca.transform(self.data)
        pl.figure()
        l = range(self.data.shape[0])
        pl.scatter(data_pca[l, 0], data_pca[l, 1],marker='x', c='r', label="data")

        #now plot the codebook
        codebook_pca = pca.transform(self.codebook)
        l2 = range(codebook_pca.shape[0])
        pl.scatter(codebook_pca[l2, 0], codebook_pca[l2, 1], c='b', linewidth=2, label="codeword")

        pl.legend()
        pl.show()




    def train(self, data):
        """
        This method trains this object with data. If the model has already been trained, then this method shall
        re-incorporate the observations into it.
        The id is the speaker id for this model. No two speaker models should be same
        and each model must have their own speaker identifier. Hence we first make sure that the id is unique for this
        model. And if not, we search the former model with this id and train it instead.
        """
        allfiles = [obj for obj in listdir(model_path) if isfile(join(model_path, obj))]
        if self.id + '.VQ' in allfiles:
            #the model already exists, we update the model instead
            filename = get_newfilename(self.id)
        else:
            filename = "%s.VQ" % self.id

        self.absolute_train(data)
        self.save(filename)

        #use this to update the HMM
        #HMM.update(filename)
        HMM_Model.adjust()


    def save(self, filename):
        """
        Saves the model in a file. It pickles the object and then saves the content, the file is reloaded by
        unpickling the file contents.
        The save filename is id.model, where id is the speaker id of this system and the model is the extension
        """
        pickle.dump(self, open(join(model_path, filename), "w"))
        return

    def calculate_score(self, data):
        """
        Given the test vectors data, this method generates a number which is the measure of how much compatible the
        data are with this model. While testing, the model with the maximum score is the winner (speaker).
        """
        #a new codebook
        testcodebook = Model.get_codebook(data)

        error = np.zeros(testcodebook.shape[0])

        #calculate the new testcodebook based on this codebook
        for index, codewords in enumerate(testcodebook):
            #the delta vector
            distance_vector = (self.codebook - codewords)**2

            #calculate the distance
            distance = np.array(map(lambda x: sum(x), distance_vector))

            error[index] += min(distance)
        return sum(error)


def get_newfilename(id):
    for i in range(1, 1000):
        new_id = "%s_%d.VQ" % (id, i)
        if not isfile(join(model_path, new_id)):
            return new_id

    raise Exception("Too many data for one ID")

def load_model(filename):
    """
    This file uses pickle to de-serialize the stored model in the Models directory and returns it as result
    raises IO exception
    """
    return pickle.load(open(join(model_path, filename),"r"))


def test(data, probabilistic=False):
    """
    This method is the heart of the testing phase. Given vectors of features, it returns the speaker id of the best
    module
    If the probabilistic parameter is specified True, then the probabilistic inference is returned in the following form
    [(id1, prob1), (id2, prob2) .....]
    Where idX is speaker id and probX is its corresponding probability
    """
    allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(join(model_path, obj)) and obj[-2:] == 'VQ']
    scores = {}
    for files in allfiles:
        model = load_model(files)
        id = files[:-6]
        scores[id] = model.calculate_score(data)

    model_score = zip(scores.keys(), scores.values())
    model_score.sort(key=lambda x: x[1])

    #WARNING: The lowest score is considered best!

    if not probabilistic:
        #just return the highest match

        return model_score[0][0]
    else:
        total_score = sum(i[1] for i in model_score)
        return [(i[0], float(i[1])/total_score) for i in model_score]
    pass
