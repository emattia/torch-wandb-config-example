## Getting started and validating your setup

To validate your Metaflow set up, navigate to the test directory:
```
cd validate-setup
```

Run the `gpu_flow.py` to validate that your deployment can run jobs on GPUs:
```bash
python gpu_flow.py --environment=pypi run --with kubernetes
```

Next we will validate you have set up Weights and Biases correctly so Metaflow tasks can use wandb.
To do this go to Outerbounds UI, and click the integrations tab. 
Click on a custom integration, and save a variable named `WANDB_API_KEY` with your key.
Then, return to the terminal and run:

```bash
python wandb_flow.py --environment=pypi run 
```

If both of these work, then proceed to the next example. 
If not we have some debugging to do. 

## End-to-end real-life example

The contents of the `end-to-end` directory show a more realistic example of multi-gpu work on Outerbounds.
You will discover these features:
- Configuring Metaflow workflows
- Configuring PyTorch scripts with Hydra, so they work cleanly with Metaflow
- Storing checkpointing's in a persistent, fault tolerant manner
- Building images for Metaflow task runtimes

```bash
cd end-to-end
```

### Training independent of Metaflow
```bash
torchrun --nproc_per_node=8 train.py 
```

### Evaluation independent of Metaflow
```bash
torchrun --nproc_per_node=2 eval.py 
```

### Training/evaluation integrated with Metaflow
```bash
python flow.py run
```

## Building custom environments

This repository also includes a template GPU image you can use to build your own images. 
```bash
docker build -t <MY_TAG> .
```
Then, push to your container registry, and specify in Metaflow workflows like
```python
    @kubernetes(..., image=<YOUR_IMAGE_FQN>, ...)
```
or via CLI:
```bash
python flow.py run --with kubernetes:image==<YOUR_IMAGE_FQN>
```# torch-wandb-config-example
