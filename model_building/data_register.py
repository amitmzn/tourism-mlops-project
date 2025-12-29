from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "amitmzn/tourism-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check/create repo
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Only upload in Colab (NOT in GitHub Actions)
# GitHub Actions sets the "CI" environment variable
if os.getenv("CI"):
    print("Running in CI/CD. Skipping local data upload.")
else:
    # Local environment (Colab) - try to upload if data exists
    data_path = "tourism_project/data"
    if os.path.isdir(data_path):
        files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        if files:
            print(f"Uploading {len(files)} file(s) from local folder...")
            api.upload_folder(
                folder_path=data_path,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print("Data uploaded successfully!")
        else:
            print("Data folder is empty. Skipping upload.")
    else:
        print("Local data folder not found. Skipping upload.")

print("Data registration complete!")
