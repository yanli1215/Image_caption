# Image_caption

In this project, a video caption system that is able to generate image captions on the keyframe and detect sound events of each story unit is developed. For image caption, a simple model, [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) based on [CLIP](https://github.com/openai/CLIP) is used, which seems to be part of a new image captioning paradigm. The objective is to focus on utilizing current models while just training a small mapping network. Rather than learning new semantic entities, this strategy simply learns to adjust the pre-trained model's current semantic knowledge towards the style of the target dataset. We use the pre-trained models trained on different datasets offered in this Github repository: https://github.com/rmokady/CLIP_prefix_caption.
The sound event detection with ASR is also implemented in this system to get more information about a video. The SED system can detect specific sound events defined [here](https://github.com/yazdayy/sound-event-detection). Evaluation metrics like BLEU-1 to BLEU-4, CIDEr, SPICE, METEOR, and ROUGE-L will be used to evaluate the caption results. 

The link to the final report is [here](https://github.com/yanli1215/Image_caption/blob/main/SCSE21_0061_Liu_Yanli.pdf).

## Set up

`$ pip install -r requirement.txt`

For PySceneDetect, we use `ffmpeg` to split video. You can download ffmpeg from: https://ffmpeg.org/download.html

Note: 
- Linux users should use a package manager (e.g. sudo apt-get install ffmpeg). 
- Windows users may require additional steps for PySceneDetect to detect ffmpeg - see the section Manually Enabling split-video Support below for details.
- macOS users can use Homebrew to install ffmpeg as below:
    -  `$ brew uninstall ffmpeg`
    -  `$ brew tap homebrew-ffmpeg/ffmpeg`
    -  `$ brew install homebrew-ffmpeg/ffmpeg/ffmpeg`
    -  `$ brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-openh264`

## Preparation
1. Move the video you want to caption into the folder named video_uploads;
2. Download the [COCO pre-trained model](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view)(Transformers), [COCO pre-trained model](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view)(MLP+finetuning), [Conceptual Caption pre-trained model](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view)(MLP+finetuning) and move it to the pretrained_models folder.

## Run
Type the following code to start captioning.
`$ python caption_video.py -i covid.mp4 --model mlp -k`

- --input, -i: The name of the input video.
- --model, -m: The type of CLIPCAP model used. The value can be either "mlp" or "transformer".
- --keepframes, -k: Keep the keyframe images after image captioning or not.
## Result
The image captioning results and sound events detected is stored in the "covid.mp4-OUTPUT-SED.json".
For more specific SED with ASR results, you can go to ./SEDwithASR/predict_results folder to see the .xml files for each story unit.

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

