'''
Python file that helps read various data type in a more efficient way

- Text data
- Image data
- Time Series data

'''
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory, text_dataset_from_directory


def imgReadSubDir(dirList, batch_size=batch_size, image_size=image_size):
    """
    argument 
      - dirList: list of subdirectories
      - batch_size
      - image_size
    return 
      - tf datasets: training, validation and/or test data, depending on the list

    """
    for d in disrList:
        return image_dataset_from_directory(directory=d,
                                            labels='inferred',
                                            label_mode='categorical',
                                            batch_size=batch_size,
                                            image_size=image_size)
    


    
def textReadSubDir(dirList, batch_size=batch_size, image_size=32):
    """
    argument 
      - dirList: list of subdirectories
      - batch_size
      - image_size
    return 
      - tf datasets: training, validation and/or test data, depending on the list

    """
    for d in disrList:
        return text_dataset_from_directory(directory=d,
                                            labels='inferred',
                                            label_mode='categorical',
                                            batch_size=batch_size
                                            )
    
    
import os
def getFileNamesList(formatta, imagePath):
    """
    argument: choose among '.jpeg', '.png', '.gif'
    return: list of files in a directory
    """
    path = imagePath

    folder = os.fsencode(path)
    
    filenames = []
    
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith( (formatta) ): 
            filenames.append(filename)
    return filenames


def readImageFromDirectory(target_size, fileNameList, nb_channel):
    """
    target_size: the 1D shape of the image to output
    fileNameList: file gotten from the directory
    nb_channel: number of channel of the input image
    
    return: 4D shape numpy array
    
    
    """
    m=len(fileNameList)
    
    ND=np.arange(m*target_size[0]*target_size[1]*nb_channel).reshape(m, target_size[0],target_size[1],nb_channel)
    
    for im in range(ND.shape[0]):
        
        img=tf.keras.preprocessing.image.load_img(
            imagePath+str(fileNameList[im]), grayscale=False, color_mode="rgb", target_size=target_size,
            interpolation="nearest")
        
        array = tf.keras.preprocessing.image.img_to_array(img)
        
        ND[im]=array
    
    return ND 


from nltk import word_tokenize

def computeMaxLenght(df, col):
    """
    having a pandas dataframe and a column of text to be converted to datasets and then be vectorized
    by keras, we need to have an approximate of the length of the sequence.
    
    arguments:df col list 
    return: an approximate maxlen
    
    """
    com = [ [x] for x in df[col].to_list() ]
    L = []
    cpt = []
    
    for k in com:
        for i in k:
            L.append(word_tokenize(i))
    for i in L:
        cpt.append(len(i))
        
    return int(np.quantile(cpt, 0.8)) 




def readStringsDfToDs(df, col):
    """
    having a dataframe and a given column containing text data, you can convert the column data to tf dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices([ [x] for x in df[col].to_list() ])
    return dataset



