
from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

try:
    blob = BlobClient(
        account_url="https://dataaiengineer.blob.core.windows.net",
        container_name="mlflow-artifacts",
        blob_name="test_upload.txt",
        credential=DefaultAzureCredential()
    )

    blob.upload_blob(b"test write", overwrite=True)
    print("Upload successful.")
except AzureError as e:
    print(f"Azure error occurred: {e}")
except Exception as ex:
    print(f"An error occurred: {ex}")