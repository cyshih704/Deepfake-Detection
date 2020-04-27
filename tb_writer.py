from tensorboardX import SummaryWriter


class TensorboardWriter():
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def tensorboard_write(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Write every epoch result on the tensorboard
        Params:
        epoch: int
        train_loss: float
        val_loss: float
        acc: float
        """
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)
        self.writer.add_scalar('val_acc', val_acc, epoch)

    def close(self):
        self.writer.close()