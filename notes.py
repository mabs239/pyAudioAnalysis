## -------------------------------------------------
## Move to working directory
cd /home/dragon_slayer/Downloads/pyAudioAnalysis2/


## Sound Directories
/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/claps/
/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/silence/
/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/noise/
/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/speechEmotion/
/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/testSounds/

## Segmentize the recorded audio
python -c '
import audioBasicIO
import audioSegmentation
[Fs, x] = audioBasicIO.readAudioFile("data/claps/clapping1.wav"); 
segments = audioSegmentation.silenceRemoval(x, Fs, 0.01, 0.01, smoothWindow = 0.5, Weight = .4, plot = True)'

python audioAnalysis.py silenceRemoval -i data/claps/clapping1.wav --smoothing 0.5 --weight 0.4 --stWin 0.01 --stStep 0.005


## Train the audio classes

python audioAnalysis.py trainClassifier -i data/silence/ data/claps/ --method svm -o data/svmClaps


python audioAnalysis.py trainClassifier -i data/silence/ data/noise/ data/claps/ --method svm -o data/svmClaps

python audioAnalysis.py trainClassifier -i data/silence/ data/noise/ data/claps/ data/cInW/ --method svm -o data/svmClaps

python audioAnalysis.py trainClassifier -i data/speechEmotion data/silence/ data/noise/ data/claps/ data/cInW/ --method svm -o data/svmClaps

## Classify the audio file
python audioAnalysis.py classifyFile -i data/testSounds/silence1.wav --model svm --classifier data/svmClaps

python audioAnalysis.py classifyFile -i data/testSounds/clap1.wav --model svm --classifier data/svmClaps

## Realtime classification
python audioAnalysisRecordAlsa.py -recordAndClassifySegments 15 out.wav data/svmClaps svm

## Learned segmentation
python audioAnalysis.py segmentClassifyFile -i data/testSounds/clapSilence.wav --model svm --modelName data/svmClaps

## -------------------------------------------------



## Clone the wiki locally
git clone https://github.com/tyiannak/pyAudioAnalysis.wiki.git


Theodoros Giannakopoulos
# GunsPath

/home/dragon_slayer/Downloads/pyAudioAnalysis-master/data/Raw Master Tracks/Master Tracks/AK-47/C_35.wav

# Segmentation
python audioAnalysis.py segmentClassifyFile -i data/scottish.wav --model svm --modelName data/svmSM
python audioAnalysis.py segmentClassifyFile -i data/sounds/recording1.wav --model svm --modelName data/svmSM

# Classify Yes No
python audioAnalysis.py trainClassifier -i data/sounds/yes/ data/sounds/no/ --method svm -o data/svmYesNo
python audioAnalysis.py segmentClassifyFile -i data/sounds/test.wav --model svm --modelName data/svmYesNo

# Silence Removel
python audioAnalysis.py silenceRemoval -i data/recording3.wav --smoothing 1.0 --weight 0.3
python audioAnalysis.py silenceRemoval -i data/sounds/recording1.wav --smoothing 1.0 --weight 0.3

python analyzeMovieSound.py


import audioSegmentation as aS
aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')

import analyzeMovieSound as gMSFF; gMSFF.getMusicSegmentsFromFile("data/scottish.wav")
import analyzeMovieSound as gMSFF; gMSFF.getMusicSegmentsFromFile("data/C35.wav").

python -c 'import analyzeMovieSound; analyzeMovieSound.getMusicSegmentsFromFile("data/scottish.wav")'
python -c 'import analyzeMovieSound; analyzeMovieSound.getMusicSegmentsFromFile("data/scottish2.wav")'
python -c 'import analyzeMovieSound; analyzeMovieSound.getMusicSegmentsFromFile("data/C35.wav")'
python -c 'import analyzeMovieSound; analyzeMovieSound.getMusicSegmentsFromFile("data/sounds/recording1.wav")'
python -c 'import analyzeMovieSound; analyzeMovieSound.getMusicSegmentsFromFile("data/sounds/GunShotYouTube.wav")'


python -c 'import audioSegmentation; audioSegmentation.mtFileClassification("data/scottish.wav", "data/svmMovies8classes", "svm", plotResults=False, gtFile="")'


python -c 'import audioSegmentation; audioSegmentation.trainHMM_fromFile("data/C35.wav","data/C35.segments","data/C35hmmModel",1,1)'
python -c 'import audioSegmentation; audioSegmentation.hmmSegmentation("data/C35.wav", "data/C35hmmModel", True, "data/C35.segments") 

python -c 'import audioSegmentation; audioSegmentation.trainHMM_fromFile("data/count.wav","data/count.segments","data/countHMMM",1,1)'
python -c 'import audioSegmentation; audioSegmentation.hmmSegmentation("data/count.wav", "data/countHMMM", True, "data/count.segments")'

