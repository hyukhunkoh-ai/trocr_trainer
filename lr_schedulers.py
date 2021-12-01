from torch.optim.lr_scheduler import LambdaLR
def get_lr(optimizer):
    lr = -1.0
    for pg in optimizer.param_groups:
        lr = max(pg["lr"], lr)
    return lr
    # decay = lr * warmup ** 0.5

def get_scheduler(optimizer,warmup=5):
    '''
    name == "inverse_sqrt"
    '''
    warmup = warmup
    init_lr = 1e-8
    lr = get_lr(optimizer)
    lr_step = (lr-init_lr) / warmup
    decay = lr * warmup ** 0.5

    def warm_decay(n):
        if n < warmup:
            return lr_step * n
        return decay * n ** -0.5

    return LambdaLR(optimizer, warm_decay)
