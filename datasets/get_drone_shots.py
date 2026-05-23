import os
import urllib.request
import tarfile

def download_and_extract():
    url = "https://zenodo.org/records/10171243/files/CROPPED_RGB.tar.gz"
    tar_filename = "CROPPED_RGB.tar.gz"
    target_dir = "drone_images"

    # 1. Download the file
    print(f"Downloading {tar_filename} from Zenodo... (This may take a moment)")
    urllib.request.urlretrieve(url, tar_filename)
    print("Download complete.")

    # 2. Create the target directory
    os.makedirs(target_dir, exist_ok=True)

    # 3. Extract the tarfile and strip the top-level directory
    print(f"Extracting images into '{target_dir}'...")
    with tarfile.open(tar_filename, "r:gz") as tar:
        for member in tar.getmembers():
            # Split the path to ignore the original 'CROPPED_RGB/' root folder
            parts = member.name.split('/', 1)
            
            # Only extract if it's a file/folder inside the root
            if len(parts) == 2 and parts[1]:
                member.name = parts[1] # Rename the internal path
                tar.extract(member, path=target_dir)

    print("Extraction complete!")

    # 4. Optional: Clean up the downloaded tar file to save space
    # os.remove(tar_filename) 

if __name__ == "__main__":
    download_and_extract()