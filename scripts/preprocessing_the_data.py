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
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
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
            "invalid_objects_count": 0,
            "fixed_clockwise_labels": 0,
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

    def save_obb_label(self, label_path, annotations):
        """Save OBB annotations to label file."""
        try:
            with open(label_path, "w") as f:
                for ann in annotations:
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    line = " ".join(
                        [str(ann[0])] + [f"{coord:.6f}" for coord in ann[1:]]
                    )
                    f.write(line + "\n")
        except Exception as e:
            print(f"Error writing label file {label_path}: {e}")

    def identify_invalid_point_counts(self):
        """Identify and move images with labels that don't have exactly 4 points."""
        print("\n=== Identifying Invalid Point Counts ===")

        invalid_object_count = {}
        invalid_point_count = {}

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

                num_of_findings = len(findings) if isinstance(findings, list) else 0

                if error_code == 1:
                    invalid_object_count[str(Path(split) / label_path.name)] = (
                        num_of_findings
                    )
                elif error_code == 9:
                    invalid_point_count[str(Path(split) / label_path.name)] = (
                        num_of_findings
                    )

        summary_file = self.invalid_points_dir / "invalid_summary.json"
        # Save invalid images for review
        summary_data = {}
        if invalid_object_count:
            summary_data["invalid_object_count"] = invalid_object_count
        if invalid_point_count:
            summary_data["invalid_point_count"] = invalid_point_count

        if summary_data:  # Only write if there's data
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            print(f"Found {len(invalid_object_count)} images with invalid object num")
            print(f"Found {len(invalid_point_count)} images with invalid point num")
            print(f"Summary saved to: {summary_file}")
        else:
            print("No images with invalid counts found")

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
                if image_file.suffix.lower() not in self.image_extensions:
                    continue

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
            f"Labels fixed for clockwise ordering: {self.stats['fixed_clockwise_labels']}"
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

    def run_preprocessing(self):
        """Run complete preprocessing pipeline."""
        print("Starting dataset preprocessing...")

        self.setup_output_directories()

        # Step 1: Identify images with invalid point counts
        self.identify_invalid_point_counts()

        # Step 2: Fix clockwise ordering (only on remaining valid files)
        self.fix_clockwise_ordering()

        # Step 3: Remove duplicates
        self.find_and_remove_duplicates()

        # Step 4: Generate summary report
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
