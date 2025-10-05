import os
import shutil
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class DocumentManager:
    """Enhanced document manager with upload, delete, and review capabilities"""

    def __init__(self, storage_dir: str = "uploaded_documents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Metadata file to track documents
        self.metadata_file = self.storage_dir / "documents_metadata.json"
        self.documents_metadata = self._load_metadata()

        # Supported file types
        self.supported_types = {
            '.pdf': 'PDF Document',
            '.txt': 'Text Document',
            '.docx': 'Word Document',
            '.doc': 'Word Document (Legacy)',
            '.md': 'Markdown Document'
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Load documents metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save documents metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate MD5 hash of file content"""
        return hashlib.md5(file_content).hexdigest()

    def _get_file_info(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        file_size = len(file_content)
        file_extension = Path(filename).suffix.lower()

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(filename)

        return {
            'filename': filename,
            'file_extension': file_extension,
            'file_size': file_size,
            'mime_type': mime_type,
            'file_hash': self._calculate_file_hash(file_content),
            'upload_timestamp': datetime.now().isoformat(),
            'readable_size': self._format_file_size(file_size)
        }

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        size = size_bytes
        unit_index = 0

        while size >= 1024 and unit_index < len(size_names) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.1f} {size_names[unit_index]}"

    def validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate uploaded file"""
        file_info = self._get_file_info(file_content, filename)

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': file_info
        }

        # Check file extension
        if file_info['file_extension'] not in self.supported_types:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Unsupported file type: {file_info['file_extension']}. "
                f"Supported types: {', '.join(self.supported_types.keys())}"
            )

        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_info['file_size'] > max_size:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"File too large: {file_info['readable_size']}. Maximum size: 50MB"
            )

        # Check for empty files
        if file_info['file_size'] == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("File is empty")

        # Check for duplicate files
        if file_info['file_hash'] in self.documents_metadata:
            existing_doc = self.documents_metadata[file_info['file_hash']]
            validation_result['warnings'].append(
                f"Similar file already exists: {existing_doc['filename']}"
            )

        # File-specific validations
        if file_info['file_extension'] == '.pdf':
            validation_result.update(self._validate_pdf(file_content))

        return validation_result

    def _validate_pdf(self, file_content: bytes) -> Dict[str, List[str]]:
        """PDF-specific validation"""
        errors = []
        warnings = []

        # Check PDF header
        if not file_content.startswith(b'%PDF-'):
            errors.append("Invalid PDF format: Missing PDF header")

        # Check for password protection (basic check)
        if b'/Encrypt' in file_content:
            errors.append("Password-protected PDFs are not supported")

        return {'errors': errors, 'warnings': warnings}

    def upload_file(self, uploaded_file, description: str = "") -> Dict[str, Any]:
        """Upload and process a file"""
        try:
            # Read file content
            file_content = uploaded_file.read()

            # Reset file pointer for potential re-reading
            uploaded_file.seek(0)

            # Validate file
            validation = self.validate_file(file_content, uploaded_file.name)

            if not validation['valid']:
                return {
                    'success': False,
                    'message': 'File validation failed',
                    'errors': validation['errors'],
                    'file_info': validation['file_info']
                }

            file_info = validation['file_info']
            file_hash = file_info['file_hash']

            # Check if file already exists
            if file_hash in self.documents_metadata:
                return {
                    'success': False,
                    'message': f"File already exists: {self.documents_metadata[file_hash]['filename']}",
                    'file_hash': file_hash,
                    'duplicate': True
                }

            # Save file to storage
            safe_filename = self._make_filename_safe(uploaded_file.name)
            file_path = self.storage_dir / f"{file_hash}_{safe_filename}"

            with open(file_path, 'wb') as f:
                f.write(file_content)

            # Update metadata
            document_metadata = {
                **file_info,
                'file_path': str(file_path),
                'description': description,
                'tags': [],
                'processed': False,
                'processing_status': 'pending',
                'error_message': None,
                'chunk_count': 0
            }

            self.documents_metadata[file_hash] = document_metadata
            self._save_metadata()

            logger.info(f"File uploaded successfully: {uploaded_file.name}")

            return {
                'success': True,
                'message': f"File '{uploaded_file.name}' uploaded successfully",
                'file_hash': file_hash,
                'file_path': str(file_path),
                'file_info': file_info,
                'warnings': validation.get('warnings', [])
            }

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return {
                'success': False,
                'message': f"Upload failed: {str(e)}",
                'error': str(e)
            }

    def _make_filename_safe(self, filename: str) -> str:
        """Make filename safe for filesystem"""
        # Remove or replace unsafe characters
        safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        safe_filename = ''.join(c for c in filename if c in safe_chars)

        # Ensure filename is not empty and not too long
        if not safe_filename or len(safe_filename) < 1:
            safe_filename = "unnamed_file"

        if len(safe_filename) > 100:
            name_part, ext_part = os.path.splitext(safe_filename)
            safe_filename = name_part[:96] + ext_part

        return safe_filename

    def delete_document(self, file_hash: str) -> Dict[str, Any]:
        """Delete a document by its hash"""
        try:
            if file_hash not in self.documents_metadata:
                return {
                    'success': False,
                    'message': 'Document not found'
                }

            document = self.documents_metadata[file_hash]
            file_path = Path(document['file_path'])

            # Delete physical file
            if file_path.exists():
                file_path.unlink()

            # Remove from metadata
            del self.documents_metadata[file_hash]
            self._save_metadata()

            logger.info(f"Document deleted: {document['filename']}")

            return {
                'success': True,
                'message': f"Document '{document['filename']}' deleted successfully"
            }

        except Exception as e:
            logger.error(f"Error deleting document {file_hash}: {e}")
            return {
                'success': False,
                'message': f"Delete failed: {str(e)}"
            }

    def get_document_list(self, sort_by: str = 'upload_timestamp', reverse: bool = True) -> List[Dict[str, Any]]:
        """Get list of all documents with metadata"""
        documents = []

        for file_hash, metadata in self.documents_metadata.items():
            doc_info = {
                'file_hash': file_hash,
                **metadata,
                'status_display': self._get_status_display(metadata)
            }
            documents.append(doc_info)

        # Sort documents
        if sort_by in ['upload_timestamp', 'file_size', 'filename']:
            try:
                documents.sort(
                    key=lambda x: x.get(sort_by, ''),
                    reverse=reverse
                )
            except Exception:
                pass  # Keep original order if sorting fails

        return documents

    def _get_status_display(self, metadata: Dict[str, Any]) -> str:
        """Get human-readable status for document"""
        if metadata.get('processed', False):
            return "✅ Processed"
        elif metadata.get('processing_status') == 'processing':
            return "🔄 Processing"
        elif metadata.get('processing_status') == 'error':
            return "❌ Error"
        else:
            return "⏳ Pending"

    def get_document_content(self, file_hash: str, max_chars: int = 1000) -> Dict[str, Any]:
        """Get preview of document content"""
        try:
            if file_hash not in self.documents_metadata:
                return {
                    'success': False,
                    'message': 'Document not found'
                }

            document = self.documents_metadata[file_hash]
            file_path = Path(document['file_path'])

            if not file_path.exists():
                return {
                    'success': False,
                    'message': 'Document file not found on disk'
                }

            # Extract content preview based on file type
            content_preview = ""
            file_extension = document['file_extension']

            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_preview = f.read(max_chars)

            elif file_extension == '.pdf':
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(file_path)
                    content_preview = ""
                    for page in reader.pages[:3]:  # First 3 pages
                        content_preview += page.extract_text()
                        if len(content_preview) >= max_chars:
                            break
                    content_preview = content_preview[:max_chars]
                except Exception as e:
                    content_preview = f"Error extracting PDF content: {str(e)}"

            elif file_extension in ['.docx', '.doc']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    content_preview = ""
                    for paragraph in doc.paragraphs:
                        content_preview += paragraph.text + "\n"
                        if len(content_preview) >= max_chars:
                            break
                    content_preview = content_preview[:max_chars]
                except Exception as e:
                    content_preview = f"Error extracting Word document content: {str(e)}"

            return {
                'success': True,
                'content_preview': content_preview,
                'document_info': document,
                'truncated': len(content_preview) >= max_chars
            }

        except Exception as e:
            logger.error(f"Error getting document content {file_hash}: {e}")
            return {
                'success': False,
                'message': f"Error reading document: {str(e)}"
            }

    def update_document_status(self, file_hash: str, status: str, chunk_count: int = 0, error_message: str = None):
        """Update document processing status"""
        if file_hash in self.documents_metadata:
            self.documents_metadata[file_hash].update({
                'processing_status': status,
                'processed': status == 'completed',
                'chunk_count': chunk_count,
                'error_message': error_message,
                'last_updated': datetime.now().isoformat()
            })
            self._save_metadata()

    def add_document_tags(self, file_hash: str, tags: List[str]):
        """Add tags to a document"""
        if file_hash in self.documents_metadata:
            existing_tags = set(self.documents_metadata[file_hash].get('tags', []))
            existing_tags.update(tags)
            self.documents_metadata[file_hash]['tags'] = list(existing_tags)
            self._save_metadata()

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by filename, description, or tags"""
        query_lower = query.lower()
        matching_docs = []

        for file_hash, metadata in self.documents_metadata.items():
            # Search in filename
            if query_lower in metadata.get('filename', '').lower():
                matching_docs.append({'file_hash': file_hash, **metadata})
                continue

            # Search in description
            if query_lower in metadata.get('description', '').lower():
                matching_docs.append({'file_hash': file_hash, **metadata})
                continue

            # Search in tags
            tags = metadata.get('tags', [])
            if any(query_lower in tag.lower() for tag in tags):
                matching_docs.append({'file_hash': file_hash, **metadata})

        return matching_docs

    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents_metadata)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        total_size = 0
        file_types = {}

        for metadata in self.documents_metadata.values():
            file_size = metadata.get('file_size', 0)
            total_size += file_size

            file_ext = metadata.get('file_extension', 'unknown')
            if file_ext not in file_types:
                file_types[file_ext] = {'count': 0, 'size': 0}

            file_types[file_ext]['count'] += 1
            file_types[file_ext]['size'] += file_size

        return {
            'total_documents': len(self.documents_metadata),
            'total_size': total_size,
            'readable_size': self._format_file_size(total_size),
            'file_types': file_types,
            'storage_directory': str(self.storage_dir)
        }

    def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """Remove orphaned files that are not in metadata"""
        try:
            tracked_files = set()
            for metadata in self.documents_metadata.values():
                tracked_files.add(Path(metadata['file_path']).name)

            # Add metadata file to tracked files
            tracked_files.add(self.metadata_file.name)

            # Find orphaned files
            orphaned_files = []
            for file_path in self.storage_dir.glob('*'):
                if file_path.is_file() and file_path.name not in tracked_files:
                    orphaned_files.append(str(file_path))

            # Remove orphaned files
            removed_count = 0
            for file_path in orphaned_files:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error removing orphaned file {file_path}: {e}")

            return {
                'success': True,
                'removed_count': removed_count,
                'orphaned_files': orphaned_files
            }

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                'success': False,
                'error': str(e)
            }