import argparse
import easydict


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default='./data/cifar-10-batches-py')
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)
    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=801)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--checkpoints', type=str, default=None)
    parser.add_argument('--pretrain', type=bool, default=True)

    args = parser.parse_args()

    return args
