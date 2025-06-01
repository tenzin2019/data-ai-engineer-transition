from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential

blob = BlobClient(
    account_url="https://dataaiengineer.blob.core.windows.net",
    container_name="mlflow-artifacts",
    blob_name="test_upload.txt",
    credential=DefaultAzureCredential()
)

blob.upload_blob(b"test blob", overwrite=True)
print("✅ Upload succeeded.")