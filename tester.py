"""
This module is associated with testing the models

USAGE:
Simply call python tester.py filename [verbose]
where, filename is the wav file, verbose [optional] if specified, produces a detailed output

RETURNS
the output file is op.txt which contains the id of the

"""

__author__ = 'pravesh'
usage = \
"""
USAGE:
Simply call python tester.py filename [verbose]
where, filename is the wav file, verbose [optional] if specified, produces a detailed output

RETURNS
the output file is op.txt which contains the id of the
"""

from sys import argv
import sys
import MFCC
import scipy.io.wavfile
import VQ
import HMM_Model


def loadWAVfile(filename):
    # Return the sample rate (in samples/sec) and data from a WAV file
    w = scipy.io.wavfile.read(filename)
    x = w[1]    # data is returned as a numpy array with a data-type determined from the file
    fs = w[0]    # Sample rate of wav file as python integer
    return x


def test(filename, verbose = False):
    rawdata = loadWAVfile(filename)
    mfcc = MFCC.extract(rawdata, show=False)

    #Test the hmm
    HMM_Model.test(mfcc, verbose)
    return


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    sys.stdout = open("stdout.txt", "w")

    if not (len(argv) == 2 or len(argv) == 3):
        print usage
        exit(0)

    if len(argv) == 3:
        test(argv[1], verbose = True)
    else:
        test(argv[1])
    #except:
    #    print sys.exc_value
    #    exit(-1)
    exit(0)




