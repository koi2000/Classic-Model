


def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr