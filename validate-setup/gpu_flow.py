from metaflow import FlowSpec, step, resources, pypi

class GPUTestFlow(FlowSpec):

    @pypi(packages={"torch": ""})
    @resources(gpu=1)
    @step
    def start(self):
        import torch

        if torch.cuda.is_available():
            print('Happy days!')
        else:
            print('Oh no, PyTorch cannot see CUDA on this machine.')
            exit()
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    GPUTestFlow()