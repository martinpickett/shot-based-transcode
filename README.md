# Shot Based Transcoder

Script to transcode a video by optimising each shot individually. Inspired by [Netflix](https://netflixtechblog.com/dynamic-optimizer-a-perceptual-video-encoding-optimization-framework-e19f1e3a277f) and [Av1an](https://github.com/master-of-zen/Av1an).


## Introduction

In film making, a film is generally made up from a series of scenes and each scene is made up from a series of shots. A simple way to think about this is that every time there is a cut, you get a new shot.

This script detects cuts in a video and uses that information to transcode the video with different setting for each shot. This allows each shot to be individually optimised for quality.

The quality metric used is VMAF (Video Multimethod Assessment Fusion), which is perceptual video quality assessment algorithm developed by Netflix ([Wikipedia](https://en.wikipedia.org/wiki/Video_Multimethod_Assessment_Fusion), [Github](https://github.com/Netflix/vmaf)). Like all computational video quality metrics VMAF is not perfect, however it is better than alternatives such as PSNR and SSIM.


## Installation

This script relies on several pieces of third party software without which it will not work.
- [Python 3](https://www.python.org)
	- [SciPy](https://www.scipy.org)
	- [Pandas](https://pandas.pydata.org)
	- [NumPy](https://numpy.org)
	- [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/)
- [FFmpeg](https://ffmpeg.org)
	- [x264](https://www.videolan.org/developers/x264.html)
	- [VMAF](https://github.com/Netflix/vmaf)

If you understand all this and are confident you have it all installed please skip ahead to usage section. Otherwise, please follow the installation guide for your operating system.

### Windows
I currently do not have a Windows PC to test with, so these instructions are approximate rather than thorough.

Firstly you need to install Python 3 and the four extra modules. In the past I have used the [Anaconda](https://www.anaconda.com/products/individual) python package on Windows. This probably comes with SciPy, NumPy and Pandas already included so you will only need to figure out how to instal PySceneDetect.

Next download the latest version of FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). I use the latest released version, but the latest git version should also work. Place the FFmpeg application in a location which allows you to call it from a command line shell (CMD, PowerShell, etc).

Finally you need to download the VMAF data models. These models should be stored in a folder near to where you intend to do your transcoding. Because of the way Netflix has implemented VMAF, on Windows you will need to specify the path to the model files and this path needs to be relative not absolute. The reason for this can be seen [here](https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md#note-about-the-model-path-on-windows).

The model files can be found in the VMAF Github repository [here](https://github.com/Netflix/vmaf/tree/master/model). You want to download `vmaf_float_v0.6.1.pkl` and `vmaf_float_v0.6.1.pkl.model`.

Warning: Netflix are in the middle of some major changes to the structure of VMAF and are changing the names of some of their files. At the moment the names given above are correct, but may not be in the future. I will do my best to keep them up to date.

With that said, you should now have all the dependencies installed on your Windows system. You now only need to download and run the Python script in this repository (see usage section for details on how to run this script).

### MacOS
I can only provide detailled installation instructions for MacOS 10.15 Catilina as that is the version I currently use. However, I expect the installation process will be similar if not identical for other modern versions of MacOS.

Firstly, Python 3 comes preinstalled on MacOS, so you only need to install the extra modules. To do that open terminal.app and run the following commands:
```shell
pip3 install scipy
pip3 install pandas
pip3 install numpy
pip3 install scenedetect
```

Next you need a version of FFmpeg with support for both x264 and VMAF. The best way I know to achieve this is to install a package manager called [Brew](https://brew.sh). The full Brew installation are on their [website](https://brew.sh), however it is very simple and I have copied them below. Once again open terminal.app and run the following command:
```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

Once you have Brew installed, you can install FFmpeg with x264 and VMAF support by opening terminal.app and running the following command:
```shell
brew install ffmpeg --with-libvmaf
```

Finally you need to download the VMAF data models and store them in the correct directory so FFmpeg can find them. To do this open terminal.app and run the following commands:
```shell
mkdir -p /usr/local/share/model
curl https://raw.githubusercontent.com/Netflix/vmaf/master/model/vmaf_float_v0.6.1.pkl.model -o /usr/local/share/model/vmaf_v0.6.1.pkl.model
curl https://raw.githubusercontent.com/Netflix/vmaf/master/model/vmaf_float_v0.6.1.pkl -o /usr/local/share/model/vmaf_v0.6.1.pkl
```

Warning: Netflix are in the middle of some major changes to the structure of VMAF and are changing the names of some of their files. Eventually this work will be done and FFmpeg will change their code to use the new file names too. For the moment the above commands work, but may not in the future. I will do my best to keep them up to date.

With that said, you should now have all the dependencies installed on your MacOS system. You now only need to download and run the Python script in this repository (see usage section for details on how to run this script).

### Linux (Ubuntu 20.04)
I can only provide detailled installation instructions for Ubuntu 20.04 as that is the flavour and version of Linux I currently use. However, I expect the installation process will be similar for other modern versions of Linux.

Firstly install Python 3 and the extra modules (Python 3 may come pre-installed with your version of Linux). To do that open a shell and run the following commands:
```shell
sudo apt install python3
pip3 install scipy
pip3 install pandas
pip3 install numpy
pip3 install scenedetect
```

Next you need a version of FFmpeg with support for both x264 and VMAF. The best way I know to achieve this is to use the version statically compiled by [John van Sickle](https://johnvansickle.com/ffmpeg/). His website has much better installation instructions then I could provide so follow his [here](https://www.johnvansickle.com/ffmpeg/faq/), pay special attention to ensuring the new version of FFmpeg is in the PATH variable. Whilst I use and recommend you use the latest released version of FFmpeg, I do not think you will have a problem with the latest git build.

Finally you need to download the VMAF data models and store them in the correct directory so FFmpeg can find them. To do this open terminal.app and run the following commands:
```shell
mkdir -p /usr/local/share/model
curl https://raw.githubusercontent.com/Netflix/vmaf/master/model/vmaf_float_v0.6.1.pkl.model -o /usr/local/share/model/vmaf_v0.6.1.pkl.model
curl https://raw.githubusercontent.com/Netflix/vmaf/master/model/vmaf_float_v0.6.1.pkl -o /usr/local/share/model/vmaf_v0.6.1.pkl
```

Warning: Netflix are in the middle of some major changes to the structure of VMAF and are changing the names of some of their files. Eventually this work will be done and FFmpeg will change their code to use the new file names too. For the moment the above commands work, but may not in the future. I will do my best to keep them up to date.

With that said, you should now have all the dependencies installed on your Linux system. You now only need to download and run the Python script in this repository (see usage section for details on how to run this script).


## Usage

Warning 1: this script will save the output with the same name as the input and in the same folder/directory that it is run from, so do not run it from the same folder/directory the  input is stored in!

Warning 2: this script is slow. Very slow. I recommend you try it on short clips first (less than a minute) so you appreciate just how slow it really is.

Basic usage:
```shell
shot-based-transcode.py path/to/input.mkv
```

You can also specify the output quality. Your eyes may differ, but I think a reasonable range of output quality to target is between 80-95. The default is 85.

To specify the output quality, use the `--quality <int>` option where the integer varies from 0 (rubbish) to 100 (perfect):
```shell
shot-based-transcode.py --quality 85 path/to/input.mkv
```

For Windows users only the model location always needs to be specified:
```shell
shot-based-transcode.py --model releative/path/to/model/vmaf_float_v0.6.1.pkl path/to/input.mkv
```

There are a few more options available which are detailed and explained in the help:
```shell
shot-based-transcode.py --help
```




## Detailed Explanation

https://netflixtechblog.com/dynamic-optimizer-a-perceptual-video-encoding-optimization-framework-e19f1e3a277f



















