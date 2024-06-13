<h2 align="center">Source-Free UDA on Remote Sensing Data</h2>
Unsupervised Domain Adaptation implementation for remote sensing dataset LoveDA where source data is unavailable

[[`Dataset`](https://doi.org/10.5281/zenodo.5706578)]

### Requirements:
- `pytorch >= 1.7.0`
- `python >= 3.6`
- `pandas >= 1.1.5`

### Getting Started
- Download the dataset and source model weights (checkpoints) from the provided link
- Generate Pseudo label by running the file generate_plabel_loveda.py
- Run train_loveda.py to train the target model on unlabeled target domain data
- Run evaluate_loveda.py and compute_iou_loveda.py to evaluate on val set and compute miou metric for each class respectively

### Reference:- 
1. Wang, J., Zheng, Z., Ma, A., Lu, X., & Zhong, Y. (2021). LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation. arXiv preprint arXiv:2110.08733.

2. Cao, Y., Zhang, H., Lu, X., Xiao, Z., Yang, K., & Wang, Y. (2024). Towards Source-free Domain Adaptive Semantic Segmentation via Importance-aware and Prototype-contrast Learning. IEEE Transactions on Intelligent Vehicles.

