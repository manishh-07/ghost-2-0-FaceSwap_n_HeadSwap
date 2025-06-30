import os
import h5py
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

def image_to_np(img_path, size=(112, 112)):
    img = Image.open(img_path).convert('RGB').resize(size)
    return np.array(img)

def save_h5_for_person(person_dir, out_dir, size=(112,112)):
    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, os.path.basename(person_dir) + '.h5')
    with h5py.File(h5_path, 'w') as f:
        imgs = np.stack([image_to_np(os.path.join(person_dir, img), size) for img in images])
        # Required fields for Voxceleb2H5Dataset
        f.create_dataset('face_arc', data=imgs)
        f.create_dataset('face_wide', data=imgs)
        # Dummy mask: all ones (white mask)
        f.create_dataset('face_wide_mask', data=np.ones_like(imgs, dtype=np.uint8))
        # Dummy segmentation: all zeros
        f.create_dataset('face_wide_parsing_segformer_B5_ce', data=np.zeros_like(imgs, dtype=np.uint8))
        # Dummy keypoints: zeros
        f.create_dataset('keypoints_68', data=np.zeros((len(imgs), 68, 2), dtype=np.int16))
        # Dummy idx_68: just sequential indices
        f.create_dataset('idx_68', data=np.arange(len(imgs)))
    print(f"Saved {h5_path}")

def convert_folder_to_h5(root, out_root):
    for person in tqdm(os.listdir(root)):
        person_dir = os.path.join(root, person)
        if os.path.isdir(person_dir):
            save_h5_for_person(person_dir, out_root)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your train or test folder (with subfolders per identity)")
    parser.add_argument("--output", type=str, required=True, help="Where to save the .h5 files")
    args = parser.parse_args()

    convert_folder_to_h5(args.input, args.output)