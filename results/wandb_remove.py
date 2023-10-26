import wandb
from datetime import datetime

# Initialize wandb API
api = wandb.Api()

institution = "lee-lab-uw-madison"
project_name = "lora-expressive-power"

# Fetch all runs of the project
runs = api.runs(f"{institution}/{project_name}")


# Define the cutoff time
cutoff_time = datetime.strptime("2023-10-25 08:00:00", "%Y-%m-%d %H:%M:%S")

# Loop through the runs and delete if they were created before the cutoff_time
for run in runs:
    created_at = run.created_at
    created_at = datetime.fromisoformat(created_at)
    if created_at < cutoff_time:
        print(f"Run created at {created_at} has been deleted!")
        run.delete()