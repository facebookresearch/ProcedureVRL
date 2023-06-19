# Dataset Preparation

## HowTo100M

Follow instructions from [dataset provider](https://www.di.ens.fr/willow/research/howto100m/) to download videos.
For the csv files of processed captions, we download from [MIL-NCE_HowTo100M](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip). After unzipping the files in a folder, set `TRAIN.TEXT` in the [config yaml files](./configs/HowTo100M) as the path to this folder.

## COIN

Follow instructions from [dataset provider](https://coin-dataset.github.io/).

## HowTo100M & COIN

We have provided data CSV files for pretraining on [full HowTo100M](./data_csv/howto100m_full), pretraining on [a subset of HowTo100M](./data_csv/howto100m_subset), [step classification on COIN](./data_csv/coin_step), [step forecasting on COIN](./data_csv/coin_next), and [recognition of procedural activities on COIN](./data_csv/coin_task).

Before you run experiments on any of them, you will need to set `DATA.PATH_PREFIX` in the [config yaml files](./configs) to the actual directory where the videos are stored. Note that due to posssibly different downloading stratergies, you may need to check the format of the video to be consistent between the data csv files and the actual videos you have.


## EPIC-Kitchens-100

Follow instructions from [dataset provider](https://github.com/epic-kitchens/epic-kitchens-100-annotations). Note that this dataset has a different dataloader structure. It loads the videos from the folder specified by `EPICKITCHENS.VISUAL_DATA_DIR` and loads the official anntoations specified by `EPICKITCHENS.ANNOTATIONS_DIR` in the [config yaml files](./configs/EK).


## Precomputed CLIP visual features
We leverage the pretrained CLIP model to align video clips and action steps in the pool, thereby creating pseudo labels for individual video clips.

Specifically, we run [CLIP (ViT-B/16)](https://github.com/openai/CLIP) to precompute visual features of video frames (1 frame per second), and save all visual features of a video into a `VIDEO_NAME.pth` file using `torch.save()`. Set `DEV.CLIP_VIS_FEAT_PATH` in the [config yaml files](./configs/HowTo100M) as the folder containing all `.pth` files.

The object saved by `torch.save()` function is as follows:
```
dct_save = {
	"clip_instances": [each element in this list is the visual feature of a frame],
	"mid_time": [each element in this list is the middle time of a frame],
    }
```