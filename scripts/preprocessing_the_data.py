import os
import shutil
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import config.settings as settings


class DatasetPreprocessor:
    def __init__(self):
        """Initialize dataset preprocessor with paths from settings."""
        self.dataset_base = Path("datasets/yolo_dataset")
        self.splits = ["train", "valid", "test"]
        self.error_code_split = ["Objects", "Points"]
        self.image_extensions = [".jpg", ".jpeg", ".png"]
        self.label_extension = ".txt"

        # Output directories for problematic data
        self.output_base = Path("preprocessing_output")
        self.invalid_points_dir = self.output_base / "invalids"
        self.duplicates_dir = self.output_base / "duplicates"

        # Filename markers for duplicate detection
        self.filename_markers = [".jpg", "_jpg", ".jpeg", "_jpeg", ".png", "_png"]

        self.stats = {
            "total_images": 0,
            "invalid_point_count": 0,
            "invalid_points_removed": 0,
            "invalid_objects_count": 0,
            "invalid_objects_removed": 0,
            "duplicates_found": 0,
            "duplicates_removed": 0,
        }

    def setup_output_directories(self):
        """Create output directories for preprocessing results."""
        self.output_base.mkdir(exist_ok=True)
        self.invalid_points_dir.mkdir(exist_ok=True)
        self.duplicates_dir.mkdir(exist_ok=True)
        print(f"Output directories created in: {self.output_base}")

    def get_label_path(self, image_path):
        """Get corresponding label file path for an image."""
        image_path = Path(image_path)
        label_dir = image_path.parent.parent / "labels"
        label_file = label_dir / f"{image_path.stem}.txt"
        return label_file

    def validate_obb_label(self, label_path):
        """
        Parse OBB label file and return if valid.
        Each annotation: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        """
        if not label_path.exists():
            return False, 0, "File Not Found"

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
                if len(lines) != 1:
                    self.stats["invalid_objects_count"] += 1
                    return False, 1, lines  # Invalid if not exactly one line

                parts = lines[0].strip().split()
                if len(parts) != 9:  # class_id + at least 8 coordinates
                    self.stats["invalid_point_count"] += 1
                    return False, 9, parts  # Invalid number of coordinates
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")

        return True, 0, parts

    def identify_invalid_point_counts(self):
        """Identify and move images with labels that don't have exactly 4 points."""
        print("\n=== Identifying Invalid Point Counts ===")

        invalid_file_count = defaultdict(list)

        for split in self.splits:
            images_dir = self.dataset_base / split / "images"
            if not images_dir.exists():
                print(f"Images directory does not exist: {images_dir}")
                continue

            for image_file in images_dir.iterdir():

                self.stats["total_images"] += 1
                label_path = self.get_label_path(image_file)
                is_valid, error_code, findings = self.validate_obb_label(label_path)

                if is_valid:
                    continue

                if error_code == 1:
                    invalid_file_count["Objects"].append(
                        {"path": image_file, "split": split}
                    )
                    self.stats["invalid_objects_count"] += 1
                elif error_code == 9:
                    invalid_file_count["Points"].append(
                        {"path": image_file, "split": split}
                    )
                    self.stats["invalid_point_count"] += 1

        if invalid_file_count:
            # Create split directories in duplicates folder
            for error_code in self.error_code_split:
                for split in self.splits:
                    split_dir = self.invalid_points_dir / error_code / split
                    split_dir.mkdir(parents=True, exist_ok=True)
                    (split_dir / "images").mkdir(parents=True, exist_ok=True)
                    (split_dir / "labels").mkdir(parents=True, exist_ok=True)

            removed_files = []

            for type, files in invalid_file_count.items():
                for file_info in files:
                    src_img = file_info["path"]
                    src_lbl = self.get_label_path(src_img)
                    split = file_info["split"]

                    # Move the file to the duplicates folder
                    dst_img = (
                        self.invalid_points_dir / type / split / "images" / src_img.name
                    )
                    dst_lbl = (
                        self.invalid_points_dir / type / split / "labels" / src_lbl.name
                    )

                    # Move image file
                    if src_img.exists():
                        shutil.move(str(src_img), str(dst_img))
                        removed_files.append(str(src_img))

                    # Move label file if it exists
                    if src_lbl.exists():
                        shutil.move(str(src_lbl), str(dst_lbl))

    def extract_filename_prefix(self, filename):
        """Extract prefix from filename before any marker."""
        filename_lower = filename.lower()

        # Find the first occurrence of any marker
        earliest_pos = len(filename)
        found_marker = None

        for marker in self.filename_markers:
            pos = filename_lower.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                found_marker = marker

        if found_marker:
            return filename[:earliest_pos]
        else:
            # No marker found, return filename without extension
            return Path(filename).stem

    def find_and_remove_duplicates(self):
        """Find and remove duplicate images based on filename prefixes."""
        print("\n=== Finding and Removing Duplicates ===")

        # Map prefixes to list of files
        prefix_to_files = defaultdict(list)

        # Collect all images
        all_images = []
        for split in self.splits:
            images_dir = self.dataset_base / split / "images"
            if not images_dir.exists():
                continue

            for image_file in images_dir.iterdir():
                prefix = self.extract_filename_prefix(image_file.name)
                prefix_to_files[prefix].append(
                    {"path": image_file, "split": split, "prefix": prefix}
                )
                all_images.append(image_file.name)

        # Find duplicates (prefixes with multiple files)
        duplicates = {}
        files_to_remove = []

        for prefix, files in prefix_to_files.items():
            if len(files) > 1:
                duplicates[prefix] = files
                self.stats["duplicates_found"] += len(files) - 1

                # Keep the first file, mark others for removal
                for file_info in files[1:]:  # Skip first file
                    files_to_remove.append(file_info)

        # Move duplicate files to review directory
        if files_to_remove:
            # Create split directories in duplicates folder
            for split in self.splits:
                split_dir = self.duplicates_dir / split
                split_dir.mkdir(exist_ok=True)
                (split_dir / "images").mkdir(exist_ok=True)
                (split_dir / "labels").mkdir(exist_ok=True)

            removed_files = []

            for file_info in files_to_remove:
                src_img = file_info["path"]
                src_lbl = self.get_label_path(src_img)
                split = file_info["split"]

                dst_img = self.duplicates_dir / split / "images" / src_img.name
                dst_lbl = self.duplicates_dir / split / "labels" / src_lbl.name

                # Move image file
                if src_img.exists():
                    shutil.move(str(src_img), str(dst_img))
                    removed_files.append(str(src_img))

                # Move label file if it exists
                if src_lbl.exists():
                    shutil.move(str(src_lbl), str(dst_lbl))

                self.stats["duplicates_removed"] += 1

            # Save list of removed files
            removed_files_log = self.duplicates_dir / "removed_files.txt"
            with open(removed_files_log, "w") as f:
                for removed_file in removed_files:
                    f.write(removed_file + "\n")

            # Save duplicate groups summary
            duplicates_summary = self.duplicates_dir / "duplicates_summary.json"
            # Convert Path objects to strings for JSON serialization
            duplicates_json = {}
            for prefix, files in duplicates.items():
                duplicates_json[prefix] = [
                    {"path": str(f["path"]), "split": f["split"], "prefix": f["prefix"]}
                    for f in files
                ]

            with open(duplicates_summary, "w") as f:
                json.dump(duplicates_json, f, indent=2)

            print(f"Found {len(duplicates)} duplicate groups")
            print(f"Removed {self.stats['duplicates_removed']} duplicate files")
            print(f"Files moved to: {self.duplicates_dir}")
            print(f"Removed files list: {removed_files_log}")
        else:
            print("No duplicates found")

    def generate_summary_report(self):
        """Generate final preprocessing summary report."""
        print("\n" + "=" * 50)
        print("DATASET PREPROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Images with invalid point counts: {self.stats['invalid_point_count']}")
        print(
            f"Images with invalid object counts: {self.stats['invalid_objects_count']}"
        )
        print(f"Duplicate groups found: {self.stats['duplicates_found']}")
        print(f"Duplicate files removed: {self.stats['duplicates_removed']}")

        final_image_count = (
            self.stats["total_images"]
            - self.stats["invalid_point_count"]
            - self.stats["duplicates_removed"]
        )
        print(f"Final clean dataset size: {final_image_count} images")

        # Save summary to file
        summary_file = self.output_base / "preprocessing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        print(f"\nDetailed summary saved to: {summary_file}")
        print(f"Review problematic files in: {self.output_base}")

    def even_split(self):
        """Split dataset into even train/val/test as 75%/15%/10% sets."""
        print("Splitting dataset into even train/val/test sets...")

        train_images_dir = self.dataset_base / "train" / "images"
        val_images_dir = self.dataset_base / "valid" / "images"
        test_images_dir = self.dataset_base / "test" / "images"
        images_for_test = []
        images_for_val = []
        train_images_dir_size = len(list(train_images_dir.iterdir()))
        val_images_dir_size = len(list(val_images_dir.iterdir()))
        test_images_dir_size = len(list(test_images_dir.iterdir()))
        total_images = (
            train_images_dir_size + val_images_dir_size + test_images_dir_size
        )
        desired_test_count = int(total_images * 0.10)
        desired_val_count = int(total_images * 0.15)

        for img in train_images_dir.iterdir():
            if test_images_dir_size < desired_test_count:
                images_for_test.append(img)
                test_images_dir_size += 1
            elif val_images_dir_size < desired_val_count:
                images_for_val.append(img)
                val_images_dir_size += 1

        for img in images_for_test:
            shutil.move(str(img), str(test_images_dir / img.name))
            lbl = self.get_label_path(img)
            if lbl.exists():
                shutil.move(str(lbl), str(test_images_dir.parent / "labels" / lbl.name))

        for img in images_for_val:
            shutil.move(str(img), str(val_images_dir / img.name))
            lbl = self.get_label_path(img)
            if lbl.exists():
                shutil.move(str(lbl), str(val_images_dir.parent / "labels" / lbl.name))

    def convert_label_from_obb_format_to_yolo_format(self, save_output_dir):
        """Convert OBB labels to YOLO format and save to specified directory."""
        print("\n=== Converting OBB Labels to YOLO Format ===")

        # Convert to Path object if it's a string
        save_output_dir = Path(save_output_dir)

        data_yaml = self.dataset_base / "data.yaml"

        for split in self.splits:
            images_dir = self.dataset_base / split / "images"
            labels_dir = self.dataset_base / split / "labels"

            # Create output directories
            output_labels_dir = save_output_dir / split / "labels"
            output_images_dir = save_output_dir / split / "images"
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            output_images_dir.mkdir(parents=True, exist_ok=True)

            if not images_dir.exists() or not labels_dir.exists():
                print(f"Skipping missing directory: {images_dir} or {labels_dir}")
                continue

            for image_file in images_dir.iterdir():
                label_file = labels_dir / f"{image_file.stem}.txt"
                if not label_file.exists():
                    continue

                try:
                    # Copy the image file
                    output_image_file = output_images_dir / image_file.name
                    shutil.copy2(image_file, output_image_file)

                    with open(label_file, "r") as f:
                        lines = f.readlines()
                        parts = lines[0].strip().split()
                        class_id = parts[0]
                        coords = list(map(float, parts[1:]))

                        # Calculate bounding box (fix coordinate extraction)
                        xs = coords[0::2]  # x coordinates: indices 0,2,4,6 from coords
                        ys = coords[1::2]  # y coordinates: indices 1,3,5,7 from coords
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # Image dimensions
                        img = cv2.imread(str(image_file))
                        h, w = img.shape[:2]

                        # Convert to YOLO format
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        box_width = x_max - x_min
                        box_height = y_max - y_min

                        yolo_label = f"{class_id} {x_center:.10f} {y_center:.10f} {box_width:.10f} {box_height:.10f}\n"

                    # Save YOLO label
                    output_label_file = output_labels_dir / f"{image_file.stem}.txt"
                    with open(output_label_file, "w") as out_f:
                        out_f.write(yolo_label)

                except Exception as e:
                    print(f"Error processing file {label_file}: {e}")

        # Copy data.yaml
        try:
            output_data_yaml = save_output_dir / "data.yaml"
            shutil.copy2(data_yaml, output_data_yaml)
            print(f"Copied data.yaml to: {output_data_yaml}")
        except Exception as e:
            print(f"Error copying data.yaml: {e}")

    def run_preprocessing(self):
        """Run complete preprocessing pipeline."""
        print("Starting dataset preprocessing...")

        self.setup_output_directories()

        # Step 1: Identify images with invalid point counts
        self.identify_invalid_point_counts()

        # Step 2: Remove duplicates
        self.find_and_remove_duplicates()

        # Step 3: Correct the splits
        self.even_split()

        # # step 4: Convert labels from OBB to YOLO format
        # self.convert_label_from_obb_format_to_yolo_format(
        #     Path(settings.CONVERTED_FROM_OBB_TO_YOLO_DIR)
        # )

        # Step 5: Generate summary report
        self.generate_summary_report()

        print("\nPreprocessing complete!")


def main():
    """Main function to run dataset preprocessing."""
    preprocessor = DatasetPreprocessor()

    try:
        preprocessor.run_preprocessing()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
