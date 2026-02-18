import wandb

# Project that the run is recorded to
project = "my-awesome-project"

# Dictionary with hyperparameters
config = {"epochs" : 1337, "lr" : 3e-4}

# The `with` syntax marks the run as finished upon exiting the `with` block,
# and it marks the run "failed" if there's an exception.
#
# In a notebook, it may be more convenient to write `run = wandb.init()`
# and manually call `run.finish()` instead of using a `with` block.
with wandb.init(project=project, config=config) as run:
    # Training code here

    # Log values to W&B with run.log()
    run.log({"accuracy": 0.9, "loss": 0.1})