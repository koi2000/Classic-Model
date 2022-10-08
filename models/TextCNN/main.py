import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
import torchtext.datasets as datasets
import jieba
import argparse
import torchtext.data
from model import TextCNN
from operation import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="TextCNN")
    parser.add_argument('-lr', type=float, default=0.001, help="initial lr")
    parser.add_argument('-epochs', type=int, default=10, help="number of epochs for train")
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-log-interval', type=int, default=100)
    parser.add_argument('-test-interval', type=int, default=200)
    parser.add_argument('-save-interval', type=int, default=1000)
    parser.add_argument('-save-dir', type=str, default='snapshot', help='directory to save the snapshot')

    parser.add_argument('-dropout', type=int, default=0.5)
    parser.add_argument('-embed-dim', type=int, default=128)
    parser.add_argument('-kernel-num', type=int, default=10)
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5')

    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-train', action='store_true', default=False, help='train a new model')
    parser.add_argument('-test', action='store_true', default=False,
                        help='test on testset, combined with -snapshot to load model')
    parser.add_argument('-predict', action='store_true', default=False, help='predict label of console input')

    args = parser.parse_args()
    return args


args = parse_arguments()


def tokenize(text):
    return [word for word in jieba.cut(text) if word.strip()]


text_field = data.Field(lower=True, tokenize=tokenize)
label_field = data.Field(sequential=False)
fields = [('text', text_field), ('label', label_field)]
train_dataset, test_dataset = data.TabularDataset.splits(
    path="./data/", format='tsv', skip_header=False,
    train='train.tsv', test='test.tsv', fields=fields
)
text_field.build_vocab(train_dataset, test_dataset, min_freq=5, max_size=50000)
label_field.build_vocab(train_dataset, test_dataset)
train_iter, test_iter = data.Iterator.splits((train_dataset, test_dataset),
                                             batch_sizes=(args.batch_size, args.batch_size),
                                             sort_key=lambda x: len(x.text))

embed_num = len(text_field.vocab)
class_num = len(label_field.vocab) - 1
kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# args.cuda = torch.cuda.is_available()
args.cuda = False

print("Parameters:")
for attr, value in sorted(args.__dict__.items()):
    print("{}={}".format(attr.upper(), value))

cnn = TextCNN(embed_num, args.embed_dim, class_num, args.kernel_num, kernel_sizes, args.dropout)
if args.snapshot is not None:
    print('Loading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print("Model parameters: " + str(pytorch_total_params))
if args.cuda:
    cnn = cnn.cuda()

if args.train:
    train(train_iter, test_iter, cnn, args)

if args.test:
    test(test_iter, cnn, args)

if args.predict:
    while (True):
        text = input(">>")
        label = predict(text, cnn, text_field, label_field, False)
        print(str(label) + " | " + text)
