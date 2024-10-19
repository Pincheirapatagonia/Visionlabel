import os
import cv2
import yaml
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import random
import logging
from sklearn.model_selection import train_test_split

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedDatasetManager:

    def __init__(self, dataset_path=None, data_yaml="data.yaml", img_size=(640, 640)):
        self.dataset_path = dataset_path
        self.data_yaml = data_yaml
        self.train_path = None
        self.val_path = None
        self.test_path = None
        self.labels = None
        self.df = None
        self.img_size = img_size
        self.dataset = {'train': {}, 'val': {}, 'test': {}}

    # 1. Load dataset folder and split it into train/val/test folders with YOLO format
    def load_media_folder(self, dataset_path, train_split=0.7, val_split=0.2, test_split=0.1, annotations_folder=None):
        """
        Load a media dataset folder, split into train/val/test, and restructure into YOLO-like folder hierarchy.
        Transfer corresponding annotations if present in the annotations folder.
        """
        self.dataset_path = dataset_path
        all_images = [f for f in os.listdir(
            self.dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Make sure annotations are tracked (YOLO formatted)
        if annotations_folder:
            annotation_files = [f.replace('.jpg', '.txt').replace(
                '.png', '.txt') for f in all_images]
        else:
            annotation_files = [None] * len(all_images)

        # Split data
        train, test, anno_train, anno_test = train_test_split(
            all_images, annotation_files, test_size=test_split, random_state=42)
        train, val, anno_train, anno_val = train_test_split(
            train, anno_train, test_size=val_split / (train_split + val_split), random_state=42)

        # Prepare directories in YOLO format
        self.train_path = os.path.join(self.dataset_path, "train/images")
        self.val_path = os.path.join(self.dataset_path, "val/images")
        self.test_path = os.path.join(self.dataset_path, "test/images")

        # Create annotation directories
        annotation_dirs = {
            'train/': anno_train,
            'val/': anno_val,
            'test/': anno_test
        }
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.dataset_path,
                        split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path,
                        split, 'labels'), exist_ok=True)

        # Move the images and corresponding annotations to their proper folders
        logging.info(
            f"Moving images and annotations to their respective folders...")
        for img, txt in zip(train, anno_train):
            self._move_to_folder(img, txt, self.train_path, os.path.join(
                self.dataset_path, 'train/labels'), annotations_folder)
        for img, txt in zip(val, anno_val):
            self._move_to_folder(img, txt, self.val_path, os.path.join(
                self.dataset_path, 'val/labels'), annotations_folder)
        for img, txt in zip(test, anno_test):
            self._move_to_folder(img, txt, self.test_path, os.path.join(
                self.dataset_path, 'test/labels'), annotations_folder)

    def _move_to_folder(self, img_file, txt_file, dest_img_folder, dest_txt_folder, annotations_folder):
        """ Move images and annotations to respective folders, if annotations exist. """
        os.rename(os.path.join(self.dataset_path, img_file),
                  os.path.join(dest_img_folder, img_file))
        if txt_file and annotations_folder:
            txt_path = os.path.join(annotations_folder, txt_file)
            if os.path.isfile(txt_path):
                os.rename(txt_path, os.path.join(dest_txt_folder, txt_file))

    # 2. Alternative loading from YOLO-style YAML file
    def load_existing_dataset(self, data_yaml):
        with open(data_yaml, 'r') as file:
            data = yaml.safe_load(file)
        self.train_path = data['train']
        self.val_path = data['val']
        self.test_path = data['test']
        self.labels = data['names']
        logging.info(f"Loaded dataset configuration from {data_yaml}.")

    # 3. Augment data with rotations, brightness, skew, etc.
    def augment_data(self, augmentation_config=None, prompt=True):
        augmentations = {}
        if isinstance(augmentation_config, str):
            # Load augmentations from YAML file
            with open(augmentation_config, 'r') as file:
                augmentations = yaml.safe_load(file)
        elif prompt:
            aug_types = ['rotation', 'skew', 'brightness', 'hue']
            for aug in aug_types:
                value = float(input(f'Enter percentage for {aug} (0-100): '))
                augmentations[aug] = value / 100.0

        self._apply_augmentations(augmentations)

    def _apply_augmentations(self, augmentations):
        # Example augmentation functions for rotation, brightness, etc.
        def rotate_image(image, angle):
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))

        for split in ['train']:
            split_img_dir = os.path.join(self.dataset_path, split, 'images')
            image_files = os.listdir(split_img_dir)
            for img_file in image_files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(split_img_dir, img_file)
                    image = cv2.imread(img_path)

                    # Rotation
                    if random.uniform(0, 1) < augmentations.get('rotation', 0):
                        angle = random.randint(-15, 15)
                        image = rotate_image(image, angle)

                    # Brightness Adjustment
                    if random.uniform(0, 1) < augmentations.get('brightness', 0):
                        pil_image = Image.fromarray(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        enhancer = ImageEnhance.Brightness(pil_image)
                        factor = random.uniform(0.8, 1.2)
                        pil_image = enhancer.enhance(factor)
                        image = cv2.cvtColor(
                            np.array(pil_image), cv2.COLOR_RGB2BGR)

                    # Could handle for hue, contrast etc. similarly
                    cv2.imwrite(img_path, image)

    # 4. Verify YOLO dataset format structure
    def verify_yolo_format(self):
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(self.dataset_path, split, 'images')
            labels_dir = os.path.join(self.dataset_path, split, 'labels')
            if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                logging.error(f"Directory structure for {split} is incorrect!")
                return False
        logging.info("YOLO dataset structure verified.")
        return True

    # 5. Normalize filenames to `dataset_train_1.jpg`, and handle labels along with them
    def normalize_filenames(self):
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.dataset_path, split, 'images')
            labels_dir = os.path.join(self.dataset_path, split, 'labels')
            for i, filename in enumerate(os.listdir(img_dir)):
                if filename.lower().endswith('.jpg'):
                    new_name = f"dataset_{split}_{i}.jpg"
                    os.rename(os.path.join(img_dir, filename),
                              os.path.join(img_dir, new_name))

                    # Annotate corresponding label
                    txt_file = filename.replace('.jpg', '.txt')
                    if os.path.exists(os.path.join(labels_dir, txt_file)):
                        os.rename(os.path.join(labels_dir, txt_file), os.path.join(
                            labels_dir, new_name.replace('.jpg', '.txt')))

    # 6. Resize images to a standard size, default set to 640x640
    def resize_images(self, size=None):
        size = size or self.img_size
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.dataset_path, split, 'images')
            for filename in os.listdir(img_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(img_dir, filename)
                    img = cv2.imread(img_path)
                    try:
                        img_resized = cv2.resize(img, size)
                        cv2.imwrite(img_path, img_resized)
                    except Exception as e:
                        logging.error(f"Error resizing {filename}: {e}")

    # 7. Generate a DataFrame with dataset file information and save it as CSV
    def generate_dataframe(self, output_csv="dataset_info.csv"):
        data = []
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.dataset_path, split, 'images')
            for filename in os.listdir(img_dir):
                img_path = os.path.join(img_dir, filename)
                pil_img = Image.open(img_path)
                width, height = pil_img.size
                data.append({'split': split, 'filename': filename,
                            'width': width, 'height': height, 'path': img_path})

        # Convert to DataFrame and save CSV
        self.df = pd.DataFrame(data)
        self.df.to_csv(output_csv, index=False)
        logging.info(f"Dataframe generated and saved to {output_csv}")

    # 8. Display metrics from the generated DataFrame
    def display_metrics(self):
        if self.df is None:
            logging.error(
                "No dataframe generated! Call `generate_dataframe()` first.")
        else:
            print(self.df.describe())

    # 9. Merge DataFrames from other datasets
    def merge_dataframes(self, other_df_path):
        try:
            other_df = pd.read_csv(other_df_path)
            self.df = pd.concat([self.df, other_df], ignore_index=True)
            logging.info("DataFrames merged successfully.")
        except Exception as e:
            logging.error(f"Error merging dataframes: {e}")

    # 10. Rename labels or merge all labels into one
    def rename_labels(self, mapping=None, merge_all=False):
        for split in ['train', 'val', 'test']:
            label_dir = os.path.join(self.dataset_path, split, 'labels')
            for label_file in os.listdir(label_dir):
                if merge_all:
                    with open(os.path.join(label_dir, label_file), 'r+') as f:
                        label_content = f.readlines()
                        # Convert all label IDs to 0 (assuming `merge_all`)
                        merged_content = [
                            '0 ' + ' '.join(line.split()[1:]) + '\n' for line in label_content]
                        f.seek(0)
                        f.writelines(merged_content)
                        f.truncate()

                elif mapping:
                    with open(os.path.join(label_dir, label_file), 'r+') as f:
                        label_content = f.readlines()
                        mapped_content = []
                        for line in label_content:
                            line_split = line.split()
                            class_id = line_split[0]
                            # Mapping class ID
                            if class_id in mapping:
                                class_id = mapping[class_id]
                            mapped_content.append(
                                f"{class_id} " + ' '.join(line_split[1:]) + '\n')
                        f.seek(0)
                        f.writelines(mapped_content)
                        f.truncate()


if __name__ == "__main__":
    
    data_path =r'C:\Users\marti\Downloads\imgs\H1'
    # Initialize
    augmentor = EnhancedDatasetManager(data_path, img_size=(640, 640))

    # Load media folder and restructure for YOLO format
    augmentor.load_media_folder(
        dataset_path=data_path)

    # Verify YOLO dataset format
    augmentor.verify_yolo_format()

    # Apply augmentations (interactively or via config file)
    augmentor.augment_data(prompt=True)

    # Resize images to 416x416
    augmentor.resize_images((416, 416))

    # Normalize filenames
    augmentor.normalize_filenames()

    # Generate DataFrame and visualize metrics
    augmentor.generate_dataframe("dataset_info.csv")
    augmentor.display_metrics()

    # Merge another CSV-based dataset
    augmentor.merge_dataframes("other_dataset_info.csv")

    # Rename labels or merge all into one
    augmentor.rename_labels(mapping={'1': '0'}, merge_all=False)
