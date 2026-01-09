# High-Spatiotemporal-Resolution Particle Imaging in Microfluidic Devices Using Event-Frame Cameras and Deep Learning
<img width="2540" height="2044" alt="workflow" src="https://github.com/user-attachments/assets/6a50a30c-719a-4ba4-92e0-0fead707f837" />


### Environment
Please download dependency packages by
```bash
pip install -r requirements.txt
```

### Dataset
Download datasets and change parameter '--dataset_path' in `Event_Frame_VFI/Timelens-XL-main/run_network.py`

Dataset and model trained weights can be downloaded from this [link]((https://doi.org/10.5281/zenodo.18193667)

### Evaluation
python run_network.py --model_pretrained Baseline.pt --skip_training

### Training from scratch
python run_network.py
