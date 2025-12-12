import os
from pathlib import Path

def count_lines_in_directory(directory_path, extensions):
    """
    Recursively counts lines in files within a directory, filtering by extension.
    Counts all lines (code, comments, and blank).
    """
    total_lines = 0
    file_count = 0
    print(f"Scanning directory: {directory_path}\n")

    # Use Path.rglob to recursively find all files
    for filepath in Path(directory_path).rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in extensions:
            try:
                # Open the file and count lines efficiently
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for line in f)
                    total_lines += line_count
                    file_count += 1
                    # print(f"{filepath.relative_to(directory_path)}: {line_count} lines")
            except Exception as e:
                print(f"Could not read {filepath}: {e}")

    print("\n--- Summary ---")
    print(f"Total files scanned: {file_count}")
    print(f"Total Lines of Code (Raw): {total_lines}")
    print(f"Extensions included: {', '.join(extensions)}")

# --- Usage ---
# Replace this with the actual path to your repository


if __name__ == "__main__":
    repo_path = Path(os.getcwd()) / "nlsn"/"nebula"/"pipelines"
    file_extensions = ['.py']
    count_lines_in_directory(repo_path, file_extensions)