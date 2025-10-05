import re
import os
import time
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    size = size_bytes
    unit_index = 0

    while size >= 1024 and unit_index < len(size_names) - 1:
        size /= 1024
        unit_index += 1

    if size < 10:
        return f"{size:.1f} {size_names[unit_index]}"
    else:
        return f"{size:.0f} {size_names[unit_index]}"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_timestamp(timestamp: str, format_type: str = 'relative') -> str:
    """Format timestamp in various formats"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = timestamp

        if format_type == 'relative':
            now = datetime.now()
            diff = now - dt

            if diff < timedelta(minutes=1):
                return "just now"
            elif diff < timedelta(hours=1):
                minutes = int(diff.total_seconds() // 60)
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            elif diff < timedelta(days=1):
                hours = int(diff.total_seconds() // 3600)
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff < timedelta(days=7):
                days = diff.days
                return f"{days} day{'s' if days != 1 else ''} ago"
            elif diff < timedelta(days=30):
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks != 1 else ''} ago"
            else:
                return dt.strftime("%Y-%m-%d")

        elif format_type == 'short':
            return dt.strftime("%m/%d %H:%M")

        elif format_type == 'long':
            return dt.strftime("%Y-%m-%d %H:%M:%S")

        elif format_type == 'date':
            return dt.strftime("%Y-%m-%d")

        else:
            return str(dt)

    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(timestamp)

def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """Truncate text to maximum length with ellipsis"""
    if len(text) <= max_length:
        return text

    return text[:max_length - len(ellipsis)].rstrip() + ellipsis

def highlight_text(text: str, query: str, highlight_class: str = "highlight") -> str:
    """Highlight search query in text"""
    if not query or not text:
        return text

    # Escape special regex characters in query
    escaped_query = re.escape(query)

    # Case-insensitive highlighting
    pattern = re.compile(f"({escaped_query})", re.IGNORECASE)

    # Replace with highlighted version
    highlighted = pattern.sub(f'<span class="{highlight_class}">\\1</span>', text)

    return highlighted

def extract_citations(text: str) -> List[str]:
    """Extract citations from text using common patterns"""
    citations = []

    # Pattern 1: [Document Title, Page X]
    pattern1 = re.compile(r'\[([^\]]+?(?:,\s*(?:Page|p\.)\s*\d+)?)\]')
    citations.extend(pattern1.findall(text))

    # Pattern 2: (Source: Document Title)
    pattern2 = re.compile(r'\(Source:\s*([^)]+)\)')
    citations.extend(pattern2.findall(text))

    # Pattern 3: [ArticleX §Y.Z]
    pattern3 = re.compile(r'\[([^]]+?\s*§[^]]+)\]')
    citations.extend(pattern3.findall(text))

    # Remove duplicates while preserving order
    unique_citations = []
    seen = set()
    for citation in citations:
        if citation not in seen:
            unique_citations.append(citation)
            seen.add(citation)

    return unique_citations

def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize filename for safe filesystem storage"""
    # Remove or replace unsafe characters
    safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    sanitized = ''.join(c for c in filename if c in safe_chars)

    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Limit length while preserving extension
    if len(sanitized) > max_length:
        name_part, ext_part = os.path.splitext(sanitized)
        available_length = max_length - len(ext_part)
        sanitized = name_part[:available_length] + ext_part

    return sanitized

def calculate_reading_time(text: str, words_per_minute: int = 200) -> Dict[str, Any]:
    """Calculate estimated reading time for text"""
    # Count words (simple word count)
    words = len(text.split())

    # Calculate reading time in minutes
    reading_time_minutes = words / words_per_minute

    # Format reading time
    if reading_time_minutes < 1:
        formatted_time = "< 1 min"
    elif reading_time_minutes < 60:
        formatted_time = f"{int(reading_time_minutes)} min"
    else:
        hours = int(reading_time_minutes // 60)
        minutes = int(reading_time_minutes % 60)
        formatted_time = f"{hours}h {minutes}m"

    return {
        'words': words,
        'minutes': reading_time_minutes,
        'formatted': formatted_time
    }

def detect_language(text: str) -> str:
    """Simple language detection based on common patterns"""
    # This is a very basic implementation
    # For production use, consider using langdetect or similar libraries

    text_lower = text.lower()

    # Common English words
    english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    english_count = sum(1 for word in english_indicators if f' {word} ' in text_lower)

    # Common Spanish words
    spanish_indicators = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo']
    spanish_count = sum(1 for word in spanish_indicators if f' {word} ' in text_lower)

    # Common French words
    french_indicators = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
    french_count = sum(1 for word in french_indicators if f' {word} ' in text_lower)

    if english_count >= spanish_count and english_count >= french_count:
        return 'en'
    elif spanish_count >= french_count:
        return 'es'
    elif french_count > 0:
        return 'fr'
    else:
        return 'unknown'

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # This is a basic implementation
    # For production use, consider using NLTK, spaCy, or similar libraries

    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'not', 'no', 'yes', 'very', 'much',
        'many', 'most', 'more', 'less', 'some', 'any', 'all', 'each', 'every',
        'from', 'into', 'onto', 'upon', 'over', 'under', 'above', 'below', 'up',
        'down', 'out', 'off', 'through', 'during', 'before', 'after', 'while',
        'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose'
    }

    # Filter out stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_words[:max_keywords]]

    return keywords

def create_text_summary(text: str, max_sentences: int = 3) -> str:
    """Create a simple extractive summary of text"""
    if not text or not text.strip():
        return ""

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    # Simple scoring based on sentence length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Score based on length (prefer medium-length sentences)
        length_score = min(len(sentence) / 100, 1.0)

        # Score based on position (prefer earlier sentences)
        position_score = 1.0 - (i / len(sentences))

        # Combined score
        total_score = (length_score * 0.7) + (position_score * 0.3)

        scored_sentences.append((sentence, total_score))

    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sentences[:max_sentences]]

    return '. '.join(top_sentences) + '.'

def validate_json(json_string: str) -> Dict[str, Any]:
    """Validate JSON string and return validation result"""
    try:
        data = json.loads(json_string)
        return {
            'valid': True,
            'data': data,
            'error': None
        }
    except json.JSONDecodeError as e:
        return {
            'valid': False,
            'data': None,
            'error': str(e)
        }

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive information about a file"""
    path = Path(file_path)

    if not path.exists():
        return {
            'exists': False,
            'error': 'File does not exist'
        }

    try:
        stat = path.stat()
        mime_type, encoding = mimetypes.guess_type(str(path))

        return {
            'exists': True,
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'mime_type': mime_type,
            'encoding': encoding,
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'parent': str(path.parent),
            'absolute_path': str(path.absolute())
        }

    except Exception as e:
        return {
            'exists': True,
            'error': str(e)
        }

def batch_process(items: List[Any], batch_size: int = 10, delay: float = 0.1):
    """Process items in batches with optional delay"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch

        if delay > 0 and i + batch_size < len(items):
            time.sleep(delay)

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry function on failure with exponential backoff"""
    def wrapper(*args, **kwargs):
        current_delay = delay

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e

                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff

    return wrapper

def create_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a text-based progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"

    filled_width = int(width * current / total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percentage = (current / total) * 100

    return f"[{bar}] {percentage:.1f}% ({current}/{total})"

def fuzzy_match(query: str, text: str, threshold: float = 0.6) -> float:
    """Simple fuzzy matching between query and text"""
    query_lower = query.lower()
    text_lower = text.lower()

    # Exact match
    if query_lower == text_lower:
        return 1.0

    # Substring match
    if query_lower in text_lower:
        return 0.8

    # Word overlap
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())

    if not query_words or not text_words:
        return 0.0

    overlap = len(query_words.intersection(text_words))
    union = len(query_words.union(text_words))

    jaccard_similarity = overlap / union if union > 0 else 0.0

    return jaccard_similarity if jaccard_similarity >= threshold else 0.0

def parse_query_parameters(query: str) -> Dict[str, Any]:
    """Parse query for special parameters and filters"""
    # This could be extended to support advanced query syntax
    # For now, it's a simple implementation

    parsed = {
        'original_query': query,
        'cleaned_query': query,
        'filters': {},
        'parameters': {}
    }

    # Look for date filters like "after:2023-01-01"
    date_pattern = r'(before|after|on):(\d{4}-\d{2}-\d{2})'
    date_matches = re.findall(date_pattern, query)
    for operator, date_str in date_matches:
        parsed['filters'][f'date_{operator}'] = date_str
        query = re.sub(f'{operator}:{date_str}', '', query)

    # Look for file type filters like "type:pdf"
    type_pattern = r'type:(\w+)'
    type_matches = re.findall(type_pattern, query)
    for file_type in type_matches:
        parsed['filters']['file_type'] = file_type
        query = re.sub(f'type:{file_type}', '', query)

    # Clean up the query
    parsed['cleaned_query'] = ' '.join(query.split())

    return parsed

class Timer:
    """Simple context manager for timing operations"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

def log_performance(operation_name: str, duration: float, details: Dict[str, Any] = None):
    """Log performance metrics for operations"""
    log_data = {
        'operation': operation_name,
        'duration': duration,
        'timestamp': datetime.now().isoformat()
    }

    if details:
        log_data.update(details)

    logger.info(f"PERFORMANCE: {operation_name} completed in {format_duration(duration)}")

    # In a production environment, this could be sent to a metrics collection system
    return log_data