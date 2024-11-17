import os

from google.cloud import storage

LIMIT = 5

# HOME_PREFIX = "/home/012392471@SJSUAD/master_project/data/waymo_1_4_3/individual_files/training"
HOME_PREFIX = "/home/012392471@SJSUAD/master_project/mmdetection3d/data/waymo/waymo_format/validation"

def get_files(bucket_name, prefix=None):
    """Lists all the blobs in the bucket or under a specific prefix."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    i = 0
    for blob in blobs:
        if i >= LIMIT:
            break

        dest = f"{HOME_PREFIX}/{blob.name.split('/')[-1]}"

        if dest[-1] == '/':
            continue

        blob.download_to_filename(dest)

        i += 1

# Example usage:
# Replace 'your-bucket-name' with your actual bucket name.
# Replace 'your/prefix/' with the folder path you want to list.
get_files('waymo_open_dataset_v_1_4_3', prefix='individual_files/validation/')