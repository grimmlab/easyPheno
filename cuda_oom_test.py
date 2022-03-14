import gc
import torch


if __name__ == '__main__':
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(100, 10000, device=device)
        for i in range(100):
            l = torch.nn.Linear(10000, 10000)
            l.to(device)
            x = l(x)

    except RuntimeError as exc:
        print(exc)
        if 'out of memory' in str(exc):
            # Recover from CUDA out of memory error
            del x, l
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print('Trial failed. Error in optim loop.')
