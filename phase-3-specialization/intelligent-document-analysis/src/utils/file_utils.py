"""
File utility functions for document processing.
"""

import hashlib
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import quote

# Try to import magic, with fallback
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


def validate_file_type(filename: str, mime_type: Optional[str] = None) -> bool:
    """
    Validate if a file type is supported.
    
    Args:
        filename: Name of the file
        mime_type: MIME type of the file (optional)
        
    Returns:
        True if file type is supported, False otherwise
    """
    # Check by MIME type if provided
    if mime_type:
        return mime_type in settings.allowed_file_types
    
    # Check by file extension
    extension = get_file_extension(filename)
    if not extension:
        return False
    
    # Map extensions to MIME types
    extension_to_mime = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.txt': 'text/plain',
    }
    
    mime_type = extension_to_mime.get(extension.lower())
    return mime_type in settings.allowed_file_types if mime_type else False


def get_file_extension(filename: str) -> Optional[str]:
    """
    Get file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension with dot (e.g., '.pdf') or None
    """
    path = Path(filename)
    return path.suffix.lower() if path.suffix else None


def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate a unique filename to avoid conflicts.
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix to add
        
    Returns:
        Unique filename
    """
    path = Path(original_filename)
    name = path.stem
    extension = path.suffix
    
    # Sanitize the filename
    name = sanitize_filename(name)
    
    # Add timestamp and random hash for uniqueness
    import time
    import random
    
    timestamp = int(time.time())
    random_hash = hashlib.md5(f"{name}{timestamp}{random.random()}".encode()).hexdigest()[:8]
    
    if prefix:
        unique_name = f"{prefix}_{name}_{timestamp}_{random_hash}{extension}"
    else:
        unique_name = f"{name}_{timestamp}_{random_hash}{extension}"
    
    return unique_name


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    # Ensure it's not empty
    if not filename:
        filename = "document"
    
    return filename


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def is_file_size_valid(file_path: str) -> bool:
    """
    Check if file size is within allowed limits.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file size is valid, False otherwise
    """
    size_mb = get_file_size_mb(file_path)
    max_size_mb = settings.max_file_size / (1024 * 1024)
    return size_mb <= max_size_mb


def is_file_size_valid_for_standard(file_path: str) -> bool:
    """
    Check if file size is within standard limits (15MB).
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file size is valid for standard upload, False otherwise
    """
    size_mb = get_file_size_mb(file_path)
    max_size_mb = settings.max_file_size_standard / (1024 * 1024)
    return size_mb <= max_size_mb


def is_file_size_valid_for_large(file_path: str) -> bool:
    """
    Check if file size is within large file limits (200MB).
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file size is valid for large upload, False otherwise
    """
    size_mb = get_file_size_mb(file_path)
    max_size_mb = settings.max_file_size_large / (1024 * 1024)
    return size_mb <= max_size_mb


def get_upload_type_for_file(file_path: str) -> str:
    """
    Determine the appropriate upload type for a file based on its size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        'standard' if file fits in 15MB limit, 'large' if it needs 200MB limit, 'too_large' if exceeds both
    """
    size_mb = get_file_size_mb(file_path)
    standard_limit_mb = settings.max_file_size_standard / (1024 * 1024)
    large_limit_mb = settings.max_file_size_large / (1024 * 1024)
    
    if size_mb <= standard_limit_mb:
        return 'standard'
    elif size_mb <= large_limit_mb:
        return 'large'
    else:
        return 'too_large'


# Streamlit file object validation functions
def get_streamlit_file_size_mb(uploaded_file) -> float:
    """
    Get file size in megabytes from a Streamlit uploaded file object.
    
    Args:
        uploaded_file: Streamlit file uploader object
        
    Returns:
        File size in MB
    """
    size_bytes = len(uploaded_file.getvalue())
    return size_bytes / (1024 * 1024)


def is_streamlit_file_size_valid_for_standard(uploaded_file) -> bool:
    """
    Check if Streamlit file size is within standard limits (15MB).
    
    Args:
        uploaded_file: Streamlit file uploader object
        
    Returns:
        True if file size is valid for standard upload, False otherwise
    """
    size_mb = get_streamlit_file_size_mb(uploaded_file)
    max_size_mb = settings.max_file_size_standard / (1024 * 1024)
    return size_mb <= max_size_mb


def is_streamlit_file_size_valid_for_large(uploaded_file) -> bool:
    """
    Check if Streamlit file size is within large file limits (200MB).
    
    Args:
        uploaded_file: Streamlit file uploader object
        
    Returns:
        True if file size is valid for large upload, False otherwise
    """
    size_mb = get_streamlit_file_size_mb(uploaded_file)
    max_size_mb = settings.max_file_size_large / (1024 * 1024)
    return size_mb <= max_size_mb


def get_streamlit_upload_type_for_file(uploaded_file) -> str:
    """
    Determine the appropriate upload type for a Streamlit file based on its size.
    
    Args:
        uploaded_file: Streamlit file uploader object
        
    Returns:
        'standard' if file fits in 15MB limit, 'large' if it needs 200MB limit, 'too_large' if exceeds both
    """
    size_mb = get_streamlit_file_size_mb(uploaded_file)
    standard_limit_mb = settings.max_file_size_standard / (1024 * 1024)
    large_limit_mb = settings.max_file_size_large / (1024 * 1024)
    
    if size_mb <= standard_limit_mb:
        return 'standard'
    elif size_mb <= large_limit_mb:
        return 'large'
    else:
        return 'too_large'


def get_mime_type(file_path: str) -> Optional[str]:
    """
    Get MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type or None if not found
    """
    # Try magic library first if available
    if MAGIC_AVAILABLE:
        try:
            return magic.from_file(file_path, mime=True)
        except Exception:
            pass
    
    # Fallback to mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """
    Get a safe filename for storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename for storage
    """
    # URL encode the filename to handle special characters
    safe_filename = quote(filename, safe='')
    
    # Replace any remaining problematic characters
    safe_filename = re.sub(r'[^\w\-_\.]', '_', safe_filename)
    
    return safe_filename


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash as hexadecimal string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def list_files_in_directory(directory_path: str, extensions: Optional[Set[str]] = None) -> List[str]:
    """
    List files in a directory with optional extension filtering.
    
    Args:
        directory_path: Path to the directory
        extensions: Set of allowed extensions (e.g., {'.pdf', '.docx'})
        
    Returns:
        List of file paths
    """
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    files = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(str(file_path))
    
    return sorted(files)


def cleanup_old_files(directory_path: str, max_age_days: int = 7) -> int:
    """
    Clean up old files from a directory.
    
    Args:
        directory_path: Path to the directory
        max_age_days: Maximum age of files in days
        
    Returns:
        Number of files deleted
    """
    import time
    
    directory = Path(directory_path)
    if not directory.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    deleted_count = 0
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # Skip files that can't be deleted
    
    return deleted_count
