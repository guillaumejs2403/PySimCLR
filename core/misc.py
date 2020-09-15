import time

class Metric_Logger():
    def __init__(self):
        self.metrics = []
        self.top1s = []
        
    def restart(self, iters):
        self.n = 0
        self.top1 = 0
        self.value = 0
        self.metric = 0
        self.iters = iters
        self.start = time.time()

    def add_metric(self, value, n=1):
        self.n += n
        self.value += value * n

    def add_metric_and_top1(self, value, top1, n):
        self.n += n
        self.value += value * n
        self.top1 += top1

    def get_mean(self):
        return self.value / self.n

    def get_acc(self):
        return self.top1 / self.n

    def print_progress(self, citer, end=False):
        if end:
            self.metrics.append(self.get_mean())
        end = '\n' if end else '\r'

        message = (f'Iter: {citer + 1} / {self.iters} | ' +
                   'Loss: {:.3f} | '.format(self.get_mean()) +
                   'Time: {:.0f}s'.format(time.time() - self.start))
        print(message, end=end)

    def print_linear_progress(self, citer, epoch=None, epochs=None, end=False):
        if end:
            self.metrics.append(self.get_mean())
            self.top1s.append(self.get_acc())
        end = '\n' if end else '\r'

        if epoch is not None and epochs is not None:
            message = f'Epoch: {epoch + 1} / {epochs} | '
        else:
            message = ''
        message = (message +
                   f'Iter: {citer + 1} / {self.iters} | ' +
                   'Loss: {:.3f} | '.format(self.get_mean()) +
                   'Top1: {:.2f} | '.format(self.get_acc() * 100) +
                   'Time: {:.0f}s'.format(time.time() - self.start))
        print(message, end=end)


def n_parameters(n_params):
    n_params = n_params / (1e6)
    print('Number of parameters: {:.3f}M'.format(n_params))