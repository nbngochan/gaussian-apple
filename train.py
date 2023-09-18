import argparse
import torch
from lightning_fabric.utilities.seed import seed_everything


seed_everything(44)

def get_args():
    parser = argparse.Argument(description='Training Object Detection Module')
    parser.add_argument(
        '--dataset', 'd', type=str, help='Root directory of dataset'
    )
    parser.add_argument(
        
    )
    pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    