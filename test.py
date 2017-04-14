import trainer
import tester
import HMM_Model
import ntpath
from os.path import isfile, join
from os import listdir

train_file_path = "./data/train"
allfiles = [join(train_file_path, obj) for obj in listdir(train_file_path)
                       if isfile(join(train_file_path, obj)) and obj[-4:] == '.wav']

for files in allfiles:
    print "Training %s as ID %s" % (files, ntpath.basename(files)[:-4])
    trainer.train(files, ntpath.basename(files)[:-4])

HMM_Model.refresh()


test_file_path = "./data/test"
allfiles = [join(test_file_path, obj) for obj in listdir(test_file_path)
                       if isfile(join(test_file_path, obj)) and obj[-4:] == '.wav']

for files in allfiles:
    print
    print
    print
    print "Testing %s" % ntpath.basename(files)[:-4]
    tester.test(files, verbose = True)
