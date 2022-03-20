# Image_caption

In this project, a video caption system that is able to generate image captions on the keyframe and detect sound events of each story unit is developed. For image caption, a simple model, [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) based on [CLIP](https://github.com/openai/CLIP) is used, which seems to be part of a new image captioning paradigm. The objective is to focus on utilizing current models while just training a small mapping network. Rather than learning new semantic entities, this strategy simply learns to adjust the pre-trained model's current semantic knowledge towards the style of the target dataset. 
The sound event detection with ASR is also implemented in this system to get more information about a video. The SED system can detect specific sound events defined [here](https://github.com/yazdayy/sound-event-detection). Evaluation metrics like BLEU-1 to BLEU-4, CIDEr, SPICE, METEOR, and ROUGE-L will be used to evaluate the caption results.

## Set up

`$ pip install -r requirement.txt`

or 

`$ pip install pandas`

`$ pip install git+https://github.com/openai/CLIP.git`

`$ pip install transformers`

`$ pip install scenedetect`

`$ pip install opencv-python==3.4.9.31`

`$ pip install scikit-image`

`$ pip install h5py`

`$ pip install librosa`

`$ pip install SpeechRecognition`

`$ pip install dicttoxml`

`$ pip install sed_eval`

$ pip install matplotlib

$ pip install prettytable`

## Preparation
Move the video you want to caption into the folder named video_uploads

## Run
Type the following code the start captioning.
`$ python caption_video.py -i covid.mp4 --model transformer -k`

- --input, -i: The path of the input video.
- --model, -m: The type of CLIPCAP model used. The value can be either "mlp" or "transformer".
- --keepframes, -k: Keep the keyframe images after image captioning or not.

## Evaluation
The evaluation of the image captioning results is based on custom reference captions. Ground truth captions should be referenced for video annotations. The annotations for every keyframe of a video are stored in a text file named 'referencen.txt' and put it into the Image_caption folder. The number of custom references has no limits. 

Firstly, make sure the Python dependencies of nlg-eval are installed. If not, run:

`$ pip install git+https://github.com/Maluuba/nlg-eval.git@master`

Then set up the nlg-eval package

 `$ nlg-eval --setup`  

Run the following command to get the evaluation metrics:

`$ nlg-eval --hypothesis=examples/hyp.txt --references=examples/ref1.txt --references=examples/ref2.txt`
- --hypothesis: The path to the files where stores the generated captions.
- --references: The path to the reference files. 

