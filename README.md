# AudioDrivenTalkingHeadGeneration

[Link to AudioStyleNet repository](https://github.com/FeliMe/AudioStyleNet)

### Modules loaded :

```bash
module load conda/2020.11-python3.8
module load cuda/11.0
module load cudnn/8.0-cuda-11.0
module load gcc/7.3.0
```

### Creating conda environment

```bash
conda env create -f environment.yml
source activate audiostylenet
```

### Usage

#### Extract data

```bash

unzip data.zip
```

#### Deepspeech

Generate audio features from audio files using deepspeech with

```bash
python run_voca_feature_extraction.py --ds_fname 0.9.3/output_graph_de.pbmm --audiofiles path/to/folder/containing/audio/files/ --out_path path/to/out/folder/
```

--audiofiles : Path to folder containing .wav audio files

--out_path : Path to output directory. It will create in it a new folder for each audio file containing one numpy file for each frame


#### Generate A from audio features

```bash
python run_audiodriven.py
```

It will use audio features located at ./data/audio_features/ to generate A at ./data/a512/ 