import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def image_to_np(img_path, size=(112, 112)):
    try:
        img = Image.open(img_path).convert('RGB').resize(size)
        return np.array(img)
    except Exception as e:
        print(f"[SKIP] Could not process {img_path}: {e}")
        return None

def save_h5_for_person(person_dir, out_dir, size=(112,112)):
    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"[WARN] No images found in {person_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, os.path.basename(person_dir) + '.h5')
    imgs = []
    valid_filenames = []
    for img in images:
        arr = image_to_np(os.path.join(person_dir, img), size)
        if arr is not None:
            imgs.append(arr)
            valid_filenames.append(img)
    if not imgs:
        print(f"[WARN] No valid images for {person_dir}")
        return
    imgs = np.stack(imgs)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('face_arc', data=imgs)
        f.create_dataset('face_wide', data=imgs)
        f.create_dataset('face_wide_mask', data=np.ones_like(imgs, dtype=np.uint8))
        f.create_dataset('face_wide_parsing_segformer_B5_ce', data=np.zeros_like(imgs, dtype=np.uint8))
        f.create_dataset('keypoints_68', data=np.zeros((len(imgs), 68, 2), dtype=np.int16))
        f.create_dataset('idx_68', data=np.arange(len(imgs)))
        f.create_dataset('filenames', data=np.string_(valid_filenames))
        f.attrs['image_size'] = size
    print(f"Saved {h5_path} ({len(imgs)} images)")

def convert_folder_to_h5(root, out_root, size=(112,112)):
    for person in tqdm(os.listdir(root)):
        person_dir = os.path.join(root, person)
        if os.path.isdir(person_dir):
            save_h5_for_person(person_dir, out_root, size=size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your train or test folder (with subfolders per identity)")
    parser.add_argument("--output", type=str, required=True, help="Where to save the .h5 files")
    parser.add_argument("--size", type=int, nargs=2, default=[112,112], help="Output image size, e.g. --size 112 112")
    args = parser.parse_args()

    convert_folder_to_h5(args.input, args.output, size=tuple(args.size))