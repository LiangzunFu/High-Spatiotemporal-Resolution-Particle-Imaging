# High-Spatiotemporal-Resolution Particle Imaging in Microfluidic Devices Using Event-Frame Cameras and Deep Learning

### Environment
Please download dependency packages by
```bash
pip install -r requirements.txt
```

### Dataset
Download datasets and change parameter '--dataset_path' in `Event_Frame_VFI/Timelens-XL-main/run_network.py`

### Dataset and Model trained weights

[link]((https://doi.org/10.5281/zenodo.18193667)

### Evaluation
python run_network.py --model_pretrained Baseline.pt --skip_training

### Training from scratch
python run_network.py
