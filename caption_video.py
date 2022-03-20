import argparse
import io
import os
import subprocess
from subprocess import check_output
import re
import time
import json
import csv
import pandas as pd
# from dicttoxml import dicttoxml
import sys
import shutil
import xml.etree.cElementTree as et
import pandas as pd
import numpy as np

start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True, help='the input video path')
parser.add_argument('--model', '-m', required=True, choices=('mlp', 'transformer'), help='type of CLIPCAP model: mlp or transformer')
parser.add_argument('--keepframes', '-k', required=False, action="store_true", help='Keep frames output or not')
args = parser.parse_args()

vidname = args.input
file = "./video_uploads/" + vidname
# file = video.mp4


####-------------------------------EXTRACT KEYFRAME---------------------------------------------------------------------------------------------------####
extractcmd = "scenedetect --min-scene-len 2s --input " + file + " detect-content --threshold 29 list-scenes save-images -n 1 -o ./frames split-video -o ./clips"
# scenedetect --min-scene-len 2s --input ./video_uploads/covid.mp4 detect-content --threshold 29 list-scenes save-images -o ./frames split-video -o ./clips
os.system(extractcmd)
# output to ./frames

current_time = time.time()
keyframe_time = current_time - start_time
previous_time = current_time

# sys.exit("Stopped")
# start_time = time.time()


####---------------------------------IMAGE CAPTION----------------------------------------------------------------------------------------------------####
# Ensure model checkpoint and wordmap in working directory
# os.system('python predict.py --img-dir "frames" --model result/model_50000 --rnn nsteplstm --max-caption-length 30 --gpu 0 --dataset-name mscoco --out prediction.json')
print("\nGenerating Captions...")

# if args.model == "mlp":
#     os.system('python caption.py --model coco --beam_size 5')
#
# if args.model == "transformer":
#     os.system('python caption_transformer.py --beam_size 5')


current_time = time.time()
caption_time = current_time - previous_time
previous_time = current_time

####------------------------------FORMATTING DATAFRAME------------------------------------------------------------------------------------------------####
f = os.path.splitext(vidname)
csvfilename = f[0] + "-Scenes.csv"
# Read in key frame timestamp data
scenes_df = pd.read_csv(csvfilename, header=1)
scenes_split_df = scenes_df[['Scene Number', 'Start Time (seconds)', 'Length (seconds)']].copy()

# Read in image caption data
path_to_current_file = os.path.realpath(__file__)
print(path_to_current_file)
current_directory = os.path.split(path_to_current_file)[0]
path_to_file = os.path.join(current_directory, "prediction.json")
with open(path_to_file) as mydata:
    prediction_dict = json.load(mydata)
# Formatting image caption data
prediction_df = pd.DataFrame(list(prediction_dict.items()), columns=['Frame', 'Caption'])
prediction_df.sort_values(by=['Frame'], inplace=True, ascending=True)
prediction_df.reset_index(inplace=True, drop=True)
# Concatenating timestamps with image captions
scenes_split_df['Caption'] = pd.Series(prediction_df['Caption'])
scenes_split_df.columns = ['frame#', 'stime', 'dur(s)', 'caption']

caption_df = scenes_split_df['caption']
caption_df.to_csv(r'image_caption.txt', header=None, index=None, sep='\t', mode='w+')

folder_loc = current_directory + "/clips/"
clip_files = [folder_loc + f for f in os.listdir(folder_loc)]
clip_files.sort()
scenes_split_df['clip location'] = clip_files
print(scenes_split_df)

current_time = time.time()
format_ic_time = current_time - previous_time
previous_time = current_time


###-----------------------------SEDwithASR--------------------------------------------------------------------------------------------------------------####
for filename in clip_files:
    actual_filename = filename[:-4]
    if filename.endswith(".mp4"):
        os.system("ffmpeg -i {0} {1}.wav".format(filename, actual_filename))
    else:
        continue

extractcmd = "python ./SEDwithASR/pytorch/predict.py predict_asr --input_dir=" + folder_loc + " --workspace=" + current_directory + "/SEDwithASR --filename='main_strong' --holdout_fold=1 --model_type='Cnn_9layers_Gru_FrameAtt' --loss_type='clip_bce' --augmentation='specaugment_mixup' --batch_size=32 --feature_type='logmel' --cuda --sample_duration=5 --overlap --overlap_value=1 --sed_thresholds --language='eng'"
os.system(extractcmd)
# output to ./frames

current_time = time.time()
SED_time = current_time - start_time
previous_time = current_time

####------------------------------FORMATTING DATAFRAME------------------------------------------------------------------------------------------------####
scenes_split_df["event"] = np.nan
prefix = current_directory + "/SEDwithASR/predict_results/covid-Scene-"
for i in scenes_split_df["frame#"]:
    loc = prefix + str(i).zfill(3) + ".xml"
    tree = et.parse(loc)
    root = tree.getroot()
    event_set = set([])
    for SoundSegment in list(root[0]):
        event_set.add(SoundSegment.text)
    scenes_split_df['event'] = scenes_split_df['event'].astype('object')
    scenes_split_df.at[i - 1, "event"] = event_set

current_time = time.time()
format_SED_time = current_time - previous_time
previous_time = current_time

###---------------------REMOVE ADJACENT DUPLICATE CAPTIONS-------------------------------------------------------------------------------------------####
##checks captions for similar adjacent captions and remove
def removeadj(threshold):
    droplist = []
    count = 0
    if threshold == 0:
        for i in range(1, len(scenes_split_df)):
            if scenes_split_df['caption'][i] == scenes_split_df['caption'][i - 1]:
                count = count + 1
                scenes_split_df.at[i - count, 'dur(s)'] = scenes_split_df['dur(s)'][i - count] + \
                                                          scenes_split_df['dur(s)'][i]
                droplist.append(i)
            else:
                count = 0
    return droplist


droplist = removeadj(0)

if droplist:
    print("\nAdjacent duplicate caption(s) found.")
    droplist_frame_num = [j + 1 for j in droplist]
    print("frame# to be dropped: ", droplist_frame_num)
    scenes_split_df.drop(scenes_split_df.index[droplist], inplace=True)
    # frame# not reset so frame images can still be easily found
    print(scenes_split_df)

current_time = time.time()
dupremove_time = current_time - previous_time
previous_time = current_time

####--------------------------OUTPUT TO JSON----------------------------------------------------------------------------------------------------------####
out = scenes_split_df.to_json(orient='records')
outputfile = vidname + '-OUTPUT-SED.json'
with open(outputfile, 'w') as f:
    f.write(out)

####-----------------------------CLEANUP--------------------------------------------------------------------------------------------------------------####
#
# deletethis = "prediction.json"
# deletealso = csvfilename
#
# ## If file exists, delete it ##
# if os.path.isfile(deletethis):
#     os.remove(deletethis)
# else:  ## Show an error ##
#     print("Error: %s file not found" % deletethis)
#
# if os.path.isfile(deletealso):
#     os.remove(deletealso)
# else:  ## Show an error ##
#     print("Error: %s file not found" % deletealso)
#
# ## removing frames
# try:
#     if args.keepframes:
#         print("Frames kept")
# except:
#     dir_path = './frames'
#     try:
#         shutil.rmtree(dir_path)
#     except OSError as e:
#         print("Error: %s : %s" % (dir_path, e.strerror))
#     print("Frames removed")
#
# # delete wav file
# for file in os.scandir(folder_loc):
#     if file.name.endswith(".wav"):
#         os.unlink(file.path)

end_time = time.time()
cleanup_time = end_time - previous_time
total_time = end_time - start_time

####-----------------------------Eval--------------------------------------------------------------------------------------------------------------####
#cmd = "nlg-eval —-hypothesis=image_caption.txt --references=reference1.txt --references=reference2.txt --references=reference3.txt --references=reference4.txt"
#os.system(cmd)

end_time = time.time()
eval_time = end_time - previous_time
total_time = end_time - start_time

print("\nTime taken(s) for...")
print("Keyframe detection: ", keyframe_time)
print("Image captioning: ", caption_time)
print("Sound event detection: ", SED_time)
print("Formatting: ", format_ic_time + format_SED_time)
print("Removing adjacent duplicates: ", dupremove_time)
print("Output and cleanup: ", cleanup_time)
print("Evaluation: ", eval_time)

print("Total time: ", total_time)
print("Output saved as: ", outputfile)
print("\n")

