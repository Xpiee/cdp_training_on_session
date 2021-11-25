'''
Script to read and extract high low segments from a session. These high low segments can be used as the calibration data instead of 
having separate high/low calibration session.

'''
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")
import time

def read_session_data(ecgPath, edaPath, ecgSampleRate, edaSampleRate, savePath):

    return


def _combine_sess2_data(dataRootPath, subID, expID, sessID, ecgSampleRate=512., edaSampleRate=128.):

    ''' Combine Session 2 ecg eda for low and high
    '''

    ecgCols = ['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL']
    edaCols = ['data_received_time', 'Timestamp', 'GSR Conductance CAL']

    subPath = os.path.join(dataRootPath, subID, expID, sessID)

    exp_1 = 'back_20211124101410' # Low cog load experiment
    exp_2 = 'back_20211124102639' # high cog load experiment

    lowSessPath = os.path.join(subPath, exp_1)   
    highSessPath = os.path.join(subPath, exp_2)

    lowecg = os.path.join(lowSessPath, 'ecg.csv')
    loweda = os.path.join(lowSessPath, 'eda.csv')

    highecg = os.path.join(highSessPath, 'ecg.csv')
    higheda = os.path.join(highSessPath, 'eda.csv')

    # Read low signals first as a pandas dataFrame

    ecgDF = pd.read_csv(lowecg, names=ecgCols)
    edaDF = pd.read_csv(loweda, names=edaCols)

    highecgDF = pd.read_csv(highecg, names=ecgCols)
    highedaDF = pd.read_csv(higheda, names=edaCols)
    
    mergedecgDF = pd.concat([ecgDF, highecgDF], axis = 0)
    mergededaDF = pd.concat([edaDF, highedaDF], axis = 0)

    mergedecgDF.reset_index(drop=True, inplace=True)
    mergededaDF.reset_index(drop=True, inplace=True)

    session_2 = 'session_2_training'

    ecgSessData = os.path.join(subPath, session_2, 'ecg.csv')
    edaSessData = os.path.join(subPath, session_2, 'eda.csv')

    mergedecgDF.to_csv(ecgSessData, index=False)
    mergededaDF.to_csv(edaSessData, index=False)

    return


if __name__ == '__main__':
    dataRootPath = f'C:/Users/Anubhav/Documents/GitHub/SignalData'
    subID = '19'
    expID = '1'
    sessID = 'A1'

    _combine_sess2_data    

# if __name__ == '__main__':

#     # send the ecg and eda path to the function

#     rootPath = f'C:/Users/Anubhav/Documents/GitHub/cdp_session_training/SignalData'

#     ecgRead = ''
#     edaRead = ''

#     ecgHz = 512.
#     edaHz = 128.

#     subjectID = 19
#     sessID = 1

#     fileTimeStamp = 'xxxxxxxxx'
#     saveMeHere = ''

#     filePrefix = 'high' # or 'low'




