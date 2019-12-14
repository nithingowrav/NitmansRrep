# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:04:21 2019

@author: Nithin_Gowrav
"""
from pydub import AudioSegment
import os


#Convert mp3 files in a directory to wav files

mp3_dir="G:/audio_source/mp3_source/"
wav_dir="G:/audio_source/wav_source/"

mp3_list=os.listdir(mp3_dir)

for i in mp3_list:
  dst = wav_dir+os.path.splitext(i)[0]+'.wav'

  # convert mp3 to wav                                                           
  sound = AudioSegment.from_mp3(mp3_dir+i)
  sound.export(dst, format="wav")


#function to dice the audio into segments based on the a given duration

def dice(self, seconds, zero_pad=False):
    """
    Cuts the AudioSegment into `seconds` segments (at most). So for example, if seconds=10,
    this will return a list of AudioSegments, in order, where each one is at most 10 seconds
    long. If `zero_pad` is True, the last item AudioSegment object will be zero padded to result
    in `seconds` seconds.
    :param seconds: The length of each segment in seconds. Can be either a float/int, in which case
                    `self.duration_seconds` / `seconds` are made, each of `seconds` length, or a
                    list-like can be given, in which case the given list must sum to
                    `self.duration_seconds` and each segment is specified by the list - e.g.
                    the 9th AudioSegment in the returned list will be `seconds[8]` seconds long.
    :param zero_pad: Whether to zero_pad the final segment if necessary. Ignored if `seconds` is
                     a list-like.
    :returns: A list of AudioSegments, each of which is the appropriate number of seconds long.
    :raises: ValueError if a list-like is given for `seconds` and the list's durations do not sum
             to `self.duration_seconds`.
    """
    try:
        total_s = sum(seconds)
        if not (self.duration_seconds <= total_s + 1 and self.duration_seconds >= total_s - 1):
            raise ValueError("`seconds` does not sum to within one second of the duration of this AudioSegment.\
                             given total seconds: %s and self.duration_seconds: %s" % (total_s, self.duration_seconds))
        starts = []
        stops = []
        time_ms = 0
        for dur in seconds:
            starts.append(time_ms)
            time_ms += dur * MS_PER_S
            stops.append(time_ms)
        zero_pad = False
    except TypeError:
        # `seconds` is not a list
        starts = range(0, int(round(self.duration_seconds * MS_PER_S)), int(round(seconds * MS_PER_S)))
        stops = (min(self.duration_seconds * MS_PER_S, start + seconds * MS_PER_S) for start in starts)
    outs = [self[start:stop] for start, stop in zip(starts, stops)]
    out_lens = [out.duration_seconds for out in outs]
    # Check if our last slice is within one ms of expected - if so, we don't need to zero pad
    if zero_pad and not (out_lens[-1] <= seconds * MS_PER_S + 1 and out_lens[-1] >= seconds * MS_PER_S - 1):
        num_zeros = self.frame_rate * (seconds * MS_PER_S - out_lens[-1])
        outs[-1] = outs[-1].zero_extend(num_samples=num_zeros)
    return outs

inputdir="G:/audio_source/wav_source/"
outdir="G:/audio_source/split_wav/"
    
for filename in os.listdir(inputdir):
    save_file_name = filename[:-4]
    myaudio = AudioSegment.from_file(inputdir+"/"+filename, "wav")
    speech = AudioSegment.from_file(inputdir+"/"+filename, "wav")
    MS_PER_S = 1000
    chunk_data=dice(speech,10)
    for i, chunk in enumerate(chunk_data):
        chunk.export(outdir+"/"+save_file_name+"_chunk{0}.wav".format(i), format="wav")
    
    
    
    