from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "amitmzn/tourism-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# Upload ONLY if the local folder exists (e.g., in Colab)
# In GitHub Actions, this block will be skipped, preventing the error.
if os.path.exists("tourism_project/data") and os.path.isdir("tourism_project/data"):
    if os.listdir("tourism_project/data"):
        print("Uploading data from local folder...")
        api.upload_folder(
            folder_path="tourism_project/data",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print("Data uploaded successfully!")
    else:
        print("Data folder is empty. Skipping upload.")
else:
    print("Local data folder not found. Skipping upload (Normal for CI/CD).")
