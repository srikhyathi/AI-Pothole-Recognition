from pathlib import Path


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path):
    """Check whether a file is a supported image."""
    return Path(path).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def get_image_files(directory):
    """Return supported image files from a directory."""
    directory = Path(directory)

    if not directory.exists():
        return []

    return [
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and is_image_file(file_path)
    ]