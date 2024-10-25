import os

from google.cloud import storage

LIMIT = 5

SJSU_ID = "<YOUR_SJSU_ID_HERE>"
HOME_PREFIX = f"/home/{SJSU_ID}@SJSUAD/master_project/data/waymo"

def get_files(bucket_name, prefix=None):
    """Lists all the blobs in the bucket or under a specific prefix."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter='/')

    # Top-level folders
    for page in blobs.pages:
        # Sub-folders
        for pre in page.prefixes:
            files = storage_client.list_blobs(bucket_name, prefix=pre, delimiter='/')

            i = 0
            for f in files:
                if i >= LIMIT:
                    break

                if f.name[-1] == '/':
                    continue
                
                download_file(f)

                i += 1

def download_file(f):
    filepath = os.path.join(HOME_PREFIX, f.name)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    f.download_to_filename(filepath)

# Example usage:
# Replace 'your-bucket-name' with your actual bucket name.
# Replace 'your/prefix/' with the folder path you want to list.
get_files('waymo_open_dataset_v_2_0_1', prefix='training/')