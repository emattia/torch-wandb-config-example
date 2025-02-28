from metaflow import FlowSpec, step, secrets, pypi

class WandbTestFlow(FlowSpec):

    @secrets(sources=["..."]) # TODO: Fill in your secret, after going to /Integrations tab on Outerbounds UI.
    @pypi(packages={"wandb": ""})
    @step
    def start(self):
        import wandb

        wandb.login(verify=True)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    WandbTestFlow()