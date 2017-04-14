"""
In addition to being a starter, this module actually shows the abstraction that we will be doing in all other
feature recognition modules. These abstractions are as follows.

Model class
Load_model method
test method

Each of these will be documented in their own method.
"""

from os import listdir, getcwd
from os.path import isfile, join
import pickle
import numpy as np
import random

#get the Model directory
model_path = join(getcwd(), "Models")

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
        self.filename = join(model_path, self.id+'.model')

    def absolute_train(self, data):
        #save the data so that we may use it some other time (e.g. visualization)
        self.data = data


    def train(self, data):
        """
        This method trains this object with data. If the model has already been trained, then this method shall
        re-incorporate the observations into it.
        The id is the speaker id for this model. No two speaker models should be same
        and each model must have their own speaker identifier. Hence we first make sure that the id is unique for this
        model. And if not, we search the former model with this id and train it instead.
        """
        allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(obj)]
        if self.id + '.model' in allfiles:
            #the model already exists, we update the model instead
            pass
        else:
            #the model doesn't exists, hence we train from scratch.
            self.absolute_train(data)

    def save(self):
        """
        Saves the model in a file. It pickles the object and then saves the content, the file is reloaded by
        unpickling the file contents.
        The save filename is id.model, where id is the speaker id of this system and the model is the extension
        """
        pickle.dump(self,  open(join(model_path, self.id+'.model'), "w"))
        return

    def calculate_score(self, data):
        """
        Given the test vectors data, this method generates a number which is the measure of how much compatible the
        data are with this model. While testing, the model with the maximum score is the winner (speaker).
        """
        pass


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
    allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(join(model_path, obj))]
    scores = {}
    for files in allfiles:
        model = load_model(files)
        id = files[:-6]
        scores[id] = model.calculate_score(data)

    model_score = zip(scores.keys(), scores.values())
    model_score.sort(key=lambda x: x[1])

    if not probabilistic:
        #just return the highest match
        return model_score[0][0]
    else:
        total_score = sum(i[1] for i in model_score)
        return [(i[0], float(i[1])/total_score) for i in model_score]
    pass
