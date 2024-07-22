# script for drawing figures, and more if needed
import torch
import numpy as np
import random
import argparse

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--method', type=str, default='LOSO',
                        help='[SD, LOSO, LOSOFT]')
    parser.add_argument('--batch_size', type=int, default=300, metavar='N',
                        help='input batch size for training (default: 300)')
    parser.add_argument('--test_batch_size', type=int, default=300, metavar='N',
                        help='input batch size for testing (default: 300)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--ft_epochs', type=int, default=150, metavar='N',
                        help='number of finetune epochs to train (default: 150)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=19, metavar='S',
                        help='random seed (default: 19)')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--model_path', type=str, default='latest_model.pt')
    
    
    args = parser.parse_args()

    if not args.no_cuda:
        use_cuda = torch.cuda.is_available()
        args.device = torch.device("cuda" if use_cuda else "cpu")

    return args