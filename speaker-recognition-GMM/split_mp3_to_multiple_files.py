# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:38:36 2019

@author: Nithin_Gowrav
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence



sound = AudioSegment.from_mp3("G:/audio_source/1/4/yes-we-can-speech.mp3")
chunks = split_on_silence(sound, 
    # must be silent for at least half a second
    min_silence_len=250,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
    

)


for i, chunk in enumerate(chunks):
    chunk.export("G:/audio_source/1/4/chunk{0}.wav".format(i), format="wav")
    
    
    
inputdir="G:/audio_source/1/4/"
mp3file="G:/audio_source/1/4/yes-we-can-speech.wav"
    
a = AudioSegment.from_mp3(mp3file) 
first_second = a[:1000] # get the first second of an mp3 
slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3
slice

for filename in os.listdir(inputdir):
    save_file_name = filename[:-4]
    myaudio = AudioSegment.from_file(inputdir+"/"+filename, "wav") 
    
import random

mylist = []

for i in range(0,100):
    x = random.randint(1,2220768)
    mylist.append(x)

mylist.sort()
start_array = mylist
duration_array = 10000
    for i in range(len(start_array)-1):
        i=0
        chunk_data = myaudio[start_array[i]:start_array[i+1]]
        chunk_data.export("G:/audio_source/1/4/"+"yes-we-can-speech"+"chunk_"+"{0}"+".wav".format(i), format="wav")
        
        
speech = AudioSegment.from_file("G:/audio_source/1/4/yes-we-can-speech.wav")


ten_seconds = 10 * 1000

for i in range(len(speech)):
    AudioSegment.dice()
    
    
    pip install audiosegment

from pydub.

first_10_seconds = speech[:ten_seconds]

last_5_seconds = song[-5000:]