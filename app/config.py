import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the RAG application"""

    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.settings = self._load_config()

        # Default configuration values
        self.defaults = {
            # Model settings
            'llm_model': 'qwen2.5:1.5b',
            'embedding_model': 'all-minilm:latest',
            'temperature': 0.1,
            'max_tokens': 2000,

            # RAG settings
            'chunk_size': 1000,
            'chunk_overlap': 50,
            'retrieval_k': 3,
            'persist_directory': 'chroma_db',

            # Document processing
            'supported_file_types': ['.pdf', '.txt', '.docx', '.doc', '.md'],
            'max_file_size_mb': 50,
            'max_files_per_upload': 10,

            # UI settings
            'theme': 'light',
            'items_per_page': 20,
            'auto_save': True,
            'show_processing_time': True,

            # System settings
            'log_level': 'INFO',
            'enable_analytics': True,
            'ollama_url': 'http://localhost:11434',
            'conversation_retention_days': 90,

            # Advanced settings
            'enable_source_citations': True,
            'enable_conversation_export': True,
            'enable_document_preview': True,
            'enable_search_highlighting': True,
            'streaming_responses': True
        }

        # Apply defaults for missing keys
        self._apply_defaults()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info("Configuration loaded from file")
                    return config
            else:
                logger.info("No config file found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return {}

    def _apply_defaults(self):
        """Apply default values for missing configuration keys"""
        for key, value in self.defaults.items():
            if key not in self.settings:
                self.settings[key] = value

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            logger.info("Configuration saved to file")
            return True
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            self.settings[key] = value
            return self.save_config()
        except Exception as e:
            logger.error(f"Error setting config value {key}: {e}")
            return False

    def update(self, updates: Dict[str, Any]) -> bool:
        """Update multiple configuration values"""
        try:
            self.settings.update(updates)
            return self.save_config()
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            self.settings = self.defaults.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting config: {e}")
            return False

    def get_model_settings(self) -> Dict[str, Any]:
        """Get model-related settings"""
        return {
            'llm_model': self.get('llm_model'),
            'embedding_model': self.get('embedding_model'),
            'temperature': self.get('temperature'),
            'max_tokens': self.get('max_tokens'),
            'ollama_url': self.get('ollama_url')
        }

    def get_rag_settings(self) -> Dict[str, Any]:
        """Get RAG-related settings"""
        return {
            'chunk_size': self.get('chunk_size'),
            'chunk_overlap': self.get('chunk_overlap'),
            'retrieval_k': self.get('retrieval_k'),
            'persist_directory': self.get('persist_directory')
        }

    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI-related settings"""
        return {
            'theme': self.get('theme'),
            'items_per_page': self.get('items_per_page'),
            'auto_save': self.get('auto_save'),
            'show_processing_time': self.get('show_processing_time')
        }

    def validate_settings(self) -> Dict[str, Any]:
        """Validate current settings and return validation results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate model settings
        if not isinstance(self.get('temperature'), (int, float)) or not (0.0 <= self.get('temperature') <= 2.0):
            validation_results['valid'] = False
            validation_results['errors'].append("Temperature must be between 0.0 and 2.0")

        if not isinstance(self.get('max_tokens'), int) or self.get('max_tokens') <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Max tokens must be a positive integer")

        # Validate RAG settings
        chunk_size = self.get('chunk_size')
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Chunk size must be a positive integer")

        chunk_overlap = self.get('chunk_overlap')
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            validation_results['valid'] = False
            validation_results['errors'].append("Chunk overlap must be non-negative and less than chunk size")

        retrieval_k = self.get('retrieval_k')
        if not isinstance(retrieval_k, int) or retrieval_k <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Retrieval K must be a positive integer")

        # Validate file settings
        max_file_size = self.get('max_file_size_mb')
        if not isinstance(max_file_size, (int, float)) or max_file_size <= 0:
            validation_results['valid'] = False
            validation_results['errors'].append("Max file size must be a positive number")

        # Check for warnings
        if chunk_overlap > chunk_size * 0.5:
            validation_results['warnings'].append("Chunk overlap is more than 50% of chunk size")

        if retrieval_k > 10:
            validation_results['warnings'].append("High retrieval K value may affect performance")

        return validation_results

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models from Ollama"""
        try:
            import requests

            # Try to connect to Ollama
            response = requests.get(f"{self.get('ollama_url')}/api/tags", timeout=5)

            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]

                # Categorize models
                llm_models = []
                embedding_models = []

                for model in available_models:
                    model_lower = model.lower()
                    if any(keyword in model_lower for keyword in ['embed', 'minilm', 'sentence']):
                        embedding_models.append(model)
                    else:
                        llm_models.append(model)

                return {
                    'llm_models': sorted(llm_models),
                    'embedding_models': sorted(embedding_models),
                    'all_models': sorted(available_models)
                }
            else:
                logger.warning(f"Failed to fetch models from Ollama: {response.status_code}")
                return {'llm_models': [], 'embedding_models': [], 'all_models': []}

        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return {'llm_models': [], 'embedding_models': [], 'all_models': []}

    def test_ollama_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama service"""
        try:
            import requests

            response = requests.get(f"{self.get('ollama_url')}/api/tags", timeout=10)

            if response.status_code == 200:
                models_data = response.json()
                model_count = len(models_data.get('models', []))

                return {
                    'connected': True,
                    'url': self.get('ollama_url'),
                    'model_count': model_count,
                    'status': 'OK'
                }
            else:
                return {
                    'connected': False,
                    'url': self.get('ollama_url'),
                    'error': f'HTTP {response.status_code}',
                    'status': 'ERROR'
                }

        except requests.exceptions.ConnectionError:
            return {
                'connected': False,
                'url': self.get('ollama_url'),
                'error': 'Connection refused - Is Ollama running?',
                'status': 'CONNECTION_ERROR'
            }
        except requests.exceptions.Timeout:
            return {
                'connected': False,
                'url': self.get('ollama_url'),
                'error': 'Connection timeout',
                'status': 'TIMEOUT'
            }
        except Exception as e:
            return {
                'connected': False,
                'url': self.get('ollama_url'),
                'error': str(e),
                'status': 'UNKNOWN_ERROR'
            }

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            'config': self.settings.copy(),
            'export_timestamp': str(datetime.now()),
            'version': '1.0'
        }

    def import_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import configuration from data"""
        try:
            if 'config' in config_data:
                imported_config = config_data['config']
            else:
                imported_config = config_data

            # Validate imported config
            validation = self._validate_imported_config(imported_config)

            if validation['valid']:
                self.settings.update(imported_config)
                self.save_config()

                return {
                    'success': True,
                    'message': 'Configuration imported successfully',
                    'warnings': validation.get('warnings', [])
                }
            else:
                return {
                    'success': False,
                    'message': 'Invalid configuration',
                    'errors': validation.get('errors', [])
                }

        except Exception as e:
            logger.error(f"Error importing config: {e}")
            return {
                'success': False,
                'message': f'Import failed: {str(e)}'
            }

    def _validate_imported_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imported configuration"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check for required keys
        required_keys = ['llm_model', 'embedding_model', 'chunk_size', 'chunk_overlap']
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            validation['errors'].append(f"Missing required keys: {', '.join(missing_keys)}")
            validation['valid'] = False

        # Type validation
        type_checks = {
            'temperature': (int, float),
            'chunk_size': int,
            'chunk_overlap': int,
            'retrieval_k': int,
            'max_file_size_mb': (int, float)
        }

        for key, expected_types in type_checks.items():
            if key in config and not isinstance(config[key], expected_types):
                validation['errors'].append(f"{key} must be of type {expected_types}")
                validation['valid'] = False

        # Check for unknown keys
        known_keys = set(self.defaults.keys())
        unknown_keys = set(config.keys()) - known_keys

        if unknown_keys:
            validation['warnings'].append(f"Unknown configuration keys will be ignored: {', '.join(unknown_keys)}")

        return validation

    # Property accessors for commonly used settings
    @property
    def llm_model(self) -> str:
        return self.get('llm_model')

    @property
    def embedding_model(self) -> str:
        return self.get('embedding_model')

    @property
    def temperature(self) -> float:
        return self.get('temperature')

    @property
    def chunk_size(self) -> int:
        return self.get('chunk_size')

    @property
    def chunk_overlap(self) -> int:
        return self.get('chunk_overlap')

    @property
    def retrieval_k(self) -> int:
        return self.get('retrieval_k')

    @property
    def persist_directory(self) -> str:
        return self.get('persist_directory')

    @property
    def max_file_size_mb(self) -> float:
        return self.get('max_file_size_mb')

    @property
    def supported_file_types(self) -> list:
        return self.get('supported_file_types')

    @property
    def items_per_page(self) -> int:
        return self.get('items_per_page')

    @property
    def theme(self) -> str:
        return self.get('theme')

    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.settings, indent=2)

# Import datetime for export timestamp
from datetime import datetime