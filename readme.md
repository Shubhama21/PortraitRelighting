

# Real-time 3D-aware Portrait Video Relighting
Official PyTorch implementation of the CVPR 2024 Highlight Paper

![Pipeline](docs/pipeline.png)

**Real-time 3D-aware Portrait Video Relighting**</br>
Ziqi Cai, Kaiwen Jiang, Shu-Yu Chen, Yu-Kun Lai, Hongbo Fu, Boxin Shi, Lin Gao*

Abstract: Synthesizing realistic videos of talking faces under custom lighting conditions and viewing angles benefits various downstream applications like video conferencing. However, most existing relighting methods are either time-consuming or unable to adjust the viewpoints. In this paper, we present the first real-time 3D-aware method for relighting in-the-wild videos of talking faces based on Neural Radiance Fields (NeRF). Given an input portrait video, our method can synthesize talking faces under both novel views and novel lighting conditions with a photo-realistic and disentangled 3D representation. Specifically, we infer an albedo tri-plane, as well as a shading tri-plane based on a desired lighting condition for each video frame with fast dual-encoders. We also leverage a temporal consistency network to ensure smooth transitions and reduce flickering artifacts. Our method runs at 32.98 fps on consumer-level hardware and achieves state-of-the-art results in terms of reconstruction quality, lighting error, lighting instability, temporal consistency and inference speed. We demonstrate the effectiveness and interactivity of our method on various portrait videos with diverse lighting and viewing conditions.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Image Relighting](#image-relighting)
  - [Video Relighting](#video-relighting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)



## Installation

### Prerequisites

The code has been tested in the following environment:
- **CUDA**: 12.4
- **PyTorch**: 2.2.2
- **Python**: 3.10.14
- **GCC**: 11.4.0
- **Ubuntu**: 22.04


### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/GhostCai/PortraitRelighting.git
    cd PortraitRelighting
    ```

2. Set up a virtual environment:

    ```bash
    conda create -n relighting python=3.10
    conda activate relighting
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. [Download](https://drive.google.com/file/d/1bYpvIJNrIdce4RIgSSv6GxDKLe0XJpEv/view?usp=drive_link) pre-trained models and save them in the `checkpoints/` directory.

5. Download NeRFFaceLighting-ffhq-64.pkl from [here](https://drive.google.com/drive/folders/1MT1aZJa0GEblJv4YUyVNi0BdwgGnQB_I) and rename it to `NeRFFaceLighting.pkl`. Save it in the `checkpoints` directory.

6. Go to the [link](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/master) to download its `BFM` directory, follow the `Prepare prerequisite models` section (we set 'model_name' as 'pretrained' ), and put its supplemented `BFM` and `checkpoints` directories under `third_party/CropPose` directory.

## Quick Start

To quickly relight an image or video, simply run the example script:

```python
python example.py
```

The relighted outputs will be saved in the examples/ directory.

## Usage

### Image Relighting

1. Initialize the Model

Set up the environment and load the necessary models:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cropposer, relighting, dpr = initialize_models(device)
example_lightings = np.load("examples/example_lightings.npy")
```

2. Warm-up the Models

Prepare the models for inference:

```python
warm_up_models(cropposer, relighting, dpr, device)
```

3. Reconstruct the Input Image (optional)

Optionally, reconstruct the input image before relighting:

```python
cropped_image, cam, planes = reconstruct_image(cropposer, relighting, dpr, device)
```

If you prefer to relight an image with a custom camera pose, you can directly modify the cam and sh parameters in the reconstruct_image function, instead of estimating them from the input image.

4. Perform Relighting

Apply relighting to the image using the pre-loaded lighting settings:

```python
perform_relighting(cropped_image, planes, relighting, device, example_lightings)
```

### Video Relighting

1. Initialize the Models

Follow the same initialization steps as for Image Relighting.

Ensure you have the following:

- Cropped video frames in examples/video/cropped/
- Corresponding camera parameters in examples/video/camera/

2. Perform Relighting on Video Frames

For a detailed implementation, refer to the `example.py` script.
## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@InProceedings{Cai_2024_CVPR,
  author    = {Cai, Ziqi and Jiang, Kaiwen and Chen, Shu-Yu and Lai, Yu-Kun and Fu, Hongbo and Shi, Boxin and Gao, Lin},
  title     = {Real-time 3D-aware Portrait Video Relighting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {6221-6231}
}
```


## License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for more details.



## Acknowledgements
This repository relies on the [NeRFFaceLighting](https://github.com/IGLICT/NeRFFaceLighting), [EG3D](https://github.com/NVlabs/eg3d), [DECA](https://github.com/yfeng95/DECA), [Deep3DRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [DPR](https://github.com/zhhoper/DPR), [E4E](https://github.com/omertov/encoder4editing), [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe), [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch/tree/master), and [facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch).


