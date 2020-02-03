# Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network
## Getting Started
1.  Create the conda environment:
    ```shell script
    conda env create -f environment.yml
    ```
    **Note**: This environment has only been tested on Windows 10 and may not work on other operating systems.
2.  Activate the conda environment:
    ```shell script
    conda activate eigen_depth_estimation
    ```

3.  Download the model weights from https://oregonstate.box.com/s/p3lbkgiwufg9rxfgx53c4svnzz2lz9av and place the file 
    in the root directory of this repo.

### Generating depth maps
1.  Run the script `demo.py`:
    ```shell script
    python demo.py -i <RGB image file or folder> -o <depth map output file or folder>
    ```
    See `python demo.py -h` for more details.
## Description
PyTorch implementation from the papers:  
https://cs.nyu.edu/~deigen/depth/depth_nips14.pdf  
https://arxiv.org/pdf/1411.4734v4.pdf


Extended model architecture and loss fn to the newer paper.  

Model Results:   
(image, ground truth depth, model prediction)
![alt text](https://raw.githubusercontent.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/master/results/model-results-1.png)


Model Arch:

<img src="https://raw.githubusercontent.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/master/results/network-architecture.png" align="center" width="600"/>


Loss Fn:

<img src="https://raw.githubusercontent.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/master/results/loss-fn.png" align="center" width="500"/>


Predictions on test image:

<img src="https://raw.githubusercontent.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/master/results/sample-image.png" align="center" width="400"/>


Pros: 
- Can detect object boundaries well, due to added image gradient component in the newer loss fn. 
- Prediction quality is decent considering from single image

Cons:
- Model produces depthmaps at lower resolution (320x240)
- Depthmaps lack clarity
- Model is really large, ~900MB, inference time is ~2s for a mini-batch of 8 (640x480) images



Model weights: https://oregonstate.box.com/s/p3lbkgiwufg9rxfgx53c4svnzz2lz9av  
NYU Depth Datasets: https://cs.nyu.edu/~silberman/datasets/
