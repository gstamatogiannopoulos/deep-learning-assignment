# Deep Learning Assignment

This repository contains the code and report for the Deep Learning assignment.

## Files
- `part1_fashion_mnist.py`: MLP and CNN experiments on Fashion-MNIST
- `part1_cifar10.py`: MLP and CNN experiments on CIFAR-10
- `part2_mura_custom_cnn.py`: custom CNN for MURA normal/abnormal study classification
- `part2_mura_transfer.py`: EfficientNetB0 transfer learning for MURA

## Install
```bash
pip install -r requirements.txt
```

## Run Part 1
```bash
python part1_fashion_mnist.py
python part1_cifar10.py
```

## Run Part 2
```bash
python part2_mura_custom_cnn.py --data_dir "/path/to/MURA-v1.1"
python part2_mura_transfer.py --data_dir "/path/to/MURA-v1.1"
```

## Notes
- Fashion-MNIST and CIFAR-10 are downloaded automatically from Keras.
- The MURA dataset is **not included** in this repository and must be downloaded separately from the official source.
- Large outputs, datasets, and trained model files are intentionally excluded from the repository.
