
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi(token=os.getenv("MLOps"))

# Use relative path for GitHub Actions compatibility
# In Colab, use the Drive path; in CI/CD, use relative "data" folder
if os.path.exists("/content/drive/MyDrive/Colab Notebooks/Predictive_Maintenance_Interim_Report/data"):
    folder_path = "/content/drive/MyDrive/Colab Notebooks/Predictive_Maintenance_Interim_Report/data"
else:
    folder_path = "data"

repo_id = "tushar77more/engine_dataset"
repo_type = "dataset"

# Check if repo exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Upload data folder
print("Files in HF repo:", api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="data",
    commit_message="Initial dataset upload"
)

print("Dataset files uploaded successfully!")
