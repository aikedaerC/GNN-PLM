import subprocess

# List of datasets
# datasets = ["devign", "draper", "reveal", "vuldeepecker", "mvd"]
datasets = ["crossvul", "cvefixes", "mvd", "diversevul"]

# Loop through each dataset and run the command
for dataset in datasets:
    command = f"python bert.py --dataset {dataset} --device cuda:2"
    try:
        # Run the command
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran for dataset: {dataset}")
    except subprocess.CalledProcessError as e:
        print(f"Error running for dataset {dataset}: {e}")
