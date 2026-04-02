# Deep Learning Assignment - Simple Code Package

This package is made to match your assignment with simple TensorFlow / Keras code.

## Folder contents

- `part1_fashion_mnist.py`
- `part1_cifar10.py`
- `part2_mura_custom_cnn.py`
- `part2_mura_transfer.py`
- `requirements.txt`
- `report/report_template.md`

## 1) Create environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## 2) Run Part 1

### Fashion-MNIST
```bash
python part1_fashion_mnist.py
```

### CIFAR-10
```bash
python part1_cifar10.py
```

The scripts will create folders:
- `outputs/fashion_mnist/`
- `outputs/cifar10/`

Inside them you will find:
- model files
- confusion matrices
- training curves
- sample predictions
- `metrics.json`

## 3) Download MURA

Go to the official Stanford AIMI / MURA pages and request the dataset.

After download and unzip, keep a folder like:

```text
MURA-v1.1/
    train/
    valid/
    train_image_paths.csv
    valid_image_paths.csv
```

If your extracted folder has a slightly different structure, just point `--data_dir` to the folder that contains the CSV files.

## 4) Run Part 2 - custom CNN

```bash
python part2_mura_custom_cnn.py --data_dir /path/to/MURA-v1.1
```

This creates:
- `outputs/mura_custom_cnn/`

## 5) Run Part 2 - transfer learning

```bash
python part2_mura_transfer.py --data_dir /path/to/MURA-v1.1
```

This creates:
- `outputs/mura_transfer/`

## 6) Write the report

Open:
- `report/report_template.md`

Copy:
- the final metric values from each `metrics.json`
- the confusion matrix screenshots
- the training curve screenshots
- a few terminal screenshots that show the code ran successfully

## Suggested order for your submission

1. Run Fashion-MNIST
2. Run CIFAR-10
3. Request / download MURA
4. Run custom CNN on MURA
5. Run transfer learning on MURA
6. Fill the report template
7. Upload this code folder to GitHub / Drive
8. Put the code link in the report
