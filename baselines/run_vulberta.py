import subprocess

# List of datasets, "devign", "draper", "reveal"
datasets = ["crossvul", "cvefixes", "mvd", "diversevul"]

# Loop through each dataset and run the command
for dataset in datasets:
    command = f"python vulberta.py --dataset {dataset} --device cuda:5"
    try:
        # Run the command
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran for dataset: {dataset}")
    except subprocess.CalledProcessError as e:
        print(f"Error running for dataset {dataset}: {e}")
