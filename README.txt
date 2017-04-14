To use the training/testing facilities, first of all, 
import the proper facility

To use Vector Quantization
#import VQ as Speech

To use VQ-HMM hybrid

#import VQHMM as Speech


To train, use the following steps
1) extract the mfccs
2)
 Create a model
model = Speech.Model("model name") 
#substitute the model name with the actual name of the Speaker, 
e.g. model = Speech.Model('Bikram')

3) Then use the train method
model.train(mfcc)



To test
1)
extract the mfcc
2) Simply use the Speech.test method
print Speech.test(mfcc)

