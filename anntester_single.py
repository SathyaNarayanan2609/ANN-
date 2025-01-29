from __future__ import division
import numpy as np
import scipy.io.wavfile as wav
from features import mfcc  # Assuming mfcc function is imported correctly
import os

class TestingNetwork:
    def __init__(self, layerSize, weights):
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        self._layerInput = []
        self._layerOutput = []
        self.weights = weights

    def forwardProc(self, input):
        InCases = input.shape[0]
        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, InCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, InCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def sgm(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1 + np.exp(-x))
        else:
            out = self.sgm(x)
            return out * (1 - out)

def testInit():
    # Setup Neural Network
    file_path = "network/vowel_network_words.npy"
    
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, "rb") as f1:
            weights = np.load(f1, allow_pickle=True, encoding='bytes')
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

    testNet = TestingNetwork((260, 25, 25, 5), weights)
    return testNet

def extractFeature(soundfile):
    # Get MFCC Feature Array
    try:
        (rate, sig) = wav.read(soundfile)
    except FileNotFoundError:
        print(f"Error: The sound file '{soundfile}' does not exist.")
        return None
    except Exception as e:
        print(f"Error reading sound file: {e}")
        return None
    
    duration = len(sig) / rate
    
    # Ensure that 'winlen' and 'winstep' are floats, but they should be positive small numbers.
    winlen = max(duration / 20, 0.025)  # Add a minimum threshold (e.g., 0.025) for window length
    winstep = max(duration / 20, 0.01)  # Add a minimum threshold for window step size

    mfcc_feat = mfcc(sig, rate, winlen=winlen, winstep=winstep)
    if mfcc_feat is None:
        print("Error: MFCC feature extraction failed.")
        return None

    print("MFCC Feature Length: " + str(len(mfcc_feat)))

    # Flatten and normalize MFCC features
    mfcc_feat_flat = mfcc_feat[:20].flatten()
    mfcc_feat_normalized = mfcc_feat_flat / np.max(np.abs(mfcc_feat_flat), axis=0)

    inputArray = np.array([mfcc_feat_normalized])
    return inputArray

def feedToNetwork(inputArray, testNet):
    if inputArray is None:
        print("Error: No input array provided to the network.")
        return None

    # Input MFCC Array to Network
    outputArray = testNet.forwardProc(inputArray)

    # Determine detected sound based on maximum output index
    indexMax = outputArray.argmax(axis=1)[0]
    print(outputArray)

    # Mapping each index to their corresponding meaning
    outStr = None
    if indexMax == 0:
        outStr = "Detected: Apple"
    elif indexMax == 1:
        outStr = "Detected: Banana"
    elif indexMax == 2:
        outStr = "Detected: Kiwi"
    elif indexMax == 3:
        outStr = "Detected: Lime"
    elif indexMax == 4:
        outStr = "Detected: Orange"

    print(outStr)
    return outStr

if __name__ == "__main__":
    testNet = testInit()
    if testNet:
        inputArray = extractFeature("test_files/test.wav")
        if inputArray is not None:
            feedToNetwork(inputArray, testNet)
