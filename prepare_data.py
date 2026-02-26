"""
Complete Data Preparation Script
==================================
Steps:
1. Delete Contagious class (only 6 images - unusable)
2. Redistribute minority classes to 80/10/10 split
3. Augment ONLY the train folders of minority classes to 200 images each

Run this ONCE before retraining.
"""

import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array, array_to_img
)

BASE_DIR = "New-Cattle-Diseasee-3"
IMG_SIZE = (224, 224)
TARGET_COUNT = 200  # target images per minority class in train
MINORITY_CLASSES = ["dermatophilosis", "pediculosis", "ringworm", "mastitis"]
random.seed(42)

# ─────────────────────────────────────────────
# STEP 1: DELETE CONTAGIOUS CLASS
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Deleting Contagious class...")
print("=" * 50)

for folder in ["train", "valid", "test"]:
    contagious_path = os.path.join(BASE_DIR, folder, "Contagious")
    if os.path.exists(contagious_path):
        shutil.rmtree(contagious_path)
        print(f"  ✅ Deleted {contagious_path}")
    else:
        print(f"  ⚠️  {contagious_path} not found, skipping")

# ─────────────────────────────────────────────
# STEP 2: REDISTRIBUTE MINORITY CLASSES 80/10/10
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Redistributing minority classes (80/10/10)...")
print("=" * 50)

for class_name in MINORITY_CLASSES:
    # Collect ALL images from train + valid + test
    all_images = []
    for folder in ["train", "valid", "test"]:
        class_path = os.path.join(BASE_DIR, folder, class_name)
        if os.path.exists(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_images.append(os.path.join(class_path, f))

    random.shuffle(all_images)
    total = len(all_images)

    # Calculate split
    n_train = int(total * 0.80)
    n_valid = int(total * 0.10)
    n_test  = total - n_train - n_valid

    # Ensure at least 5 in valid and test
    if n_valid < 5:
        n_valid = 5
    if n_test < 5:
        n_test = 5
    n_train = total - n_valid - n_test

    splits = {
        "train": all_images[:n_train],
        "valid": all_images[n_train:n_train + n_valid],
        "test":  all_images[n_train + n_valid:]
    }

    # Clear old files and copy redistributed ones
    for folder, images in splits.items():
        class_path = os.path.join(BASE_DIR, folder, class_name)
        os.makedirs(class_path, exist_ok=True)

        # Clear existing
        for f in os.listdir(class_path):
            file_path = os.path.join(class_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Copy new split
        for img_path in images:
            filename = os.path.basename(img_path)
            # Avoid overwriting if same filename from different folder
            dest_path = os.path.join(class_path, filename)
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_copy{ext}"
                dest_path = os.path.join(class_path, filename)
            shutil.copy2(img_path, dest_path)

    print(f"  ✅ {class_name}: {total} total → train={n_train}, valid={n_valid}, test={n_test}")

# ─────────────────────────────────────────────
# STEP 3: AUGMENT TRAIN FOLDERS ONLY
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Augmenting minority class train folders...")
print(f"        Target: {TARGET_COUNT} images per class")
print("=" * 50)

# Heavy augmentation - simulates real-world cattle photo variations
augmentor = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

for class_name in MINORITY_CLASSES:
    train_class_dir = os.path.join(BASE_DIR, "train", class_name)

    if not os.path.exists(train_class_dir):
        print(f"  ⚠️  {class_name} train folder not found, skipping")
        continue

    # Get original images only (not previously augmented ones)
    existing = [
        f for f in os.listdir(train_class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        and not f.startswith('aug_')  # skip already augmented
    ]
    current_count = len(existing)
    images_needed = max(0, TARGET_COUNT - current_count)

    print(f"\n  📁 {class_name}: {current_count} original images → need {images_needed} more")

    if images_needed == 0:
        print(f"     Already at target, skipping.")
        continue

    # Load original images
    images = []
    for img_file in existing:
        img_path = os.path.join(train_class_dir, img_file)
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            images.append(img_to_array(img))
        except Exception as e:
            print(f"     ⚠️  Could not load {img_file}: {e}")

    if not images:
        print(f"     ❌ No valid images found, skipping.")
        continue

    images_array = np.array(images)
    generated = 0
    img_index = 0

    while generated < images_needed:
        original = images_array[img_index % len(images_array)]
        original = original.reshape((1,) + original.shape)

        for aug_batch in augmentor.flow(original, batch_size=1):
            aug_img = array_to_img(aug_batch[0])
            save_name = f"aug_{class_name}_{generated:04d}.jpg"
            aug_img.save(os.path.join(train_class_dir, save_name))
            generated += 1
            break

        img_index += 1

        if generated % 50 == 0:
            print(f"     Generated {generated}/{images_needed}...")

    final_count = len(os.listdir(train_class_dir))
    print(f"     ✅ Done! {class_name} train now has {final_count} images")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)
print(f"{'Class':<20} {'Train':>8} {'Valid':>8} {'Test':>8}")
print("-" * 50)

for class_name in sorted(os.listdir(os.path.join(BASE_DIR, "train"))):
    counts = []
    for folder in ["train", "valid", "test"]:
        path = os.path.join(BASE_DIR, folder, class_name)
        count = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(path) else 0
        counts.append(count)
    print(f"{class_name:<20} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}")

print("\n🎉 Data preparation complete!")
print("   Next steps:")
print("   1. Retrain your model from scratch")
print("   2. Add class_weight=class_weight_dict to model.fit()")
