import wandb

def init_wandb(project_name="vqvae-genomics", 
               run_name=None,
               config=None):
    """
    Initialize a wandb run
    
    Args:
        project_name (str): Name of the wandb project
        run_name (str, optional): Specific name for this run
        config (dict, optional): Configuration parameters to track
    """
    if config is None:
        config = {
            "architecture": "VQ-VAE",
            "dataset": "viral-sequences",
            "vocab_size": 4097,
            "num_codes": 512,
            "code_dim": 64,
            "embed_dim": 128,
            "hidden_dim": 256,
            "batch_size": 64,
            "learning_rate": 2e-4,
            "num_epochs": 5,
            "max_seq_length": 150
        }
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True  # Allow multiple runs in the same process
    )
    
    return run

def log_metrics(metrics_dict, step=None):
    """
    Log metrics to wandb
    
    Args:
        metrics_dict (dict): Dictionary of metrics to log
        step (int, optional): Step number for logging
    """
    wandb.log(metrics_dict, step=step)

def finish_run():
    """
    Cleanup and finish the wandb run
    """
    wandb.finish()

# Example usage:
if __name__ == "__main__":
    # Initialize wandb
    run = init_wandb(
        project_name="my-project",
        run_name="experiment-1",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    )
    
    # Example training loop
    for epoch in range(10):
        # Your training code here
        
        # Log metrics
        log_metrics({
            "epoch": epoch,
            "loss": 0.5 - epoch * 0.05,  # Example metrics
            "accuracy": 0.8 + epoch * 0.02
        })
    
    # Finish the run
    finish_run()