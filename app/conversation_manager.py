import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manager for conversation history and persistence"""

    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Conversation index file
        self.index_file = self.storage_dir / "conversations_index.json"
        self.conversations_index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load conversations index"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading conversation index: {e}")
            return {}

    def _save_index(self):
        """Save conversations index"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversation index: {e}")

    def create_conversation(self, title: str = None) -> str:
        """Create a new conversation and return its ID"""
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        conversation_data = {
            'id': conversation_id,
            'title': title,
            'created_at': timestamp,
            'updated_at': timestamp,
            'messages': [],
            'metadata': {
                'total_messages': 0,
                'total_tokens': 0,
                'average_response_time': 0,
                'tags': [],
                'bookmarked': False
            }
        }

        # Save to index
        self.conversations_index[conversation_id] = {
            'id': conversation_id,
            'title': title,
            'created_at': timestamp,
            'updated_at': timestamp,
            'message_count': 0,
            'file_path': f"conversation_{conversation_id}.json"
        }

        # Save conversation file
        self._save_conversation(conversation_id, conversation_data)
        self._save_index()

        logger.info(f"Created new conversation: {conversation_id}")
        return conversation_id

    def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to a conversation"""
        try:
            conversation = self.load_conversation(conversation_id)
            if not conversation:
                logger.error(f"Conversation {conversation_id} not found")
                return False

            # Add message with timestamp and ID
            message_with_metadata = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'processing_time': message.get('processing_time', 0),
                **message
            }

            conversation['messages'].append(message_with_metadata)

            # Update metadata
            conversation['updated_at'] = datetime.now().isoformat()
            conversation['metadata']['total_messages'] = len(conversation['messages'])

            # Calculate average response time for assistant messages
            assistant_messages = [m for m in conversation['messages'] if m.get('role') == 'assistant']
            if assistant_messages:
                total_time = sum(m.get('processing_time', 0) for m in assistant_messages)
                conversation['metadata']['average_response_time'] = total_time / len(assistant_messages)

            # Update index
            if conversation_id in self.conversations_index:
                self.conversations_index[conversation_id].update({
                    'updated_at': conversation['updated_at'],
                    'message_count': len(conversation['messages'])
                })

            # Save conversation
            self._save_conversation(conversation_id, conversation)
            self._save_index()

            return True

        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            return False

    def _save_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]):
        """Save conversation to file"""
        try:
            file_path = self.storage_dir / f"conversation_{conversation_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {e}")

    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation by ID"""
        try:
            if conversation_id not in self.conversations_index:
                return None

            file_path = self.storage_dir / f"conversation_{conversation_id}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None

        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
            return None

    def get_conversation_list(self, limit: int = 50, sort_by: str = 'updated_at') -> List[Dict[str, Any]]:
        """Get list of conversations with metadata"""
        conversations = []

        for conv_id, conv_info in self.conversations_index.items():
            conversations.append({
                **conv_info,
                'summary': self._get_conversation_summary(conv_id)
            })

        # Sort conversations
        reverse = sort_by in ['updated_at', 'created_at', 'message_count']
        try:
            conversations.sort(
                key=lambda x: x.get(sort_by, ''),
                reverse=reverse
            )
        except Exception:
            pass  # Keep original order if sorting fails

        return conversations[:limit]

    def _get_conversation_summary(self, conversation_id: str) -> str:
        """Get a summary of the conversation"""
        try:
            conversation = self.load_conversation(conversation_id)
            if not conversation or not conversation.get('messages'):
                return "No messages"

            # Get first user message as summary
            user_messages = [m for m in conversation['messages'] if m.get('role') == 'user']
            if user_messages:
                first_message = user_messages[0].get('content', '')
                if len(first_message) > 100:
                    return first_message[:97] + "..."
                return first_message

            return f"{len(conversation['messages'])} messages"

        except Exception as e:
            logger.error(f"Error getting summary for conversation {conversation_id}: {e}")
            return "Error loading summary"

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        try:
            # Remove from index
            if conversation_id in self.conversations_index:
                del self.conversations_index[conversation_id]
                self._save_index()

            # Delete conversation file
            file_path = self.storage_dir / f"conversation_{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()

            logger.info(f"Deleted conversation: {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False

    def search_conversations(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by title or content"""
        query_lower = query.lower()
        matching_conversations = []

        for conv_id, conv_info in self.conversations_index.items():
            # Search in title
            if query_lower in conv_info.get('title', '').lower():
                conversation = self.load_conversation(conv_id)
                if conversation:
                    matching_conversations.append({
                        **conv_info,
                        'match_type': 'title',
                        'summary': self._get_conversation_summary(conv_id)
                    })
                continue

            # Search in message content
            conversation = self.load_conversation(conv_id)
            if conversation:
                for message in conversation.get('messages', []):
                    content = message.get('content', '')
                    if query_lower in content.lower():
                        matching_conversations.append({
                            **conv_info,
                            'match_type': 'content',
                            'match_preview': self._get_match_preview(content, query),
                            'summary': self._get_conversation_summary(conv_id)
                        })
                        break

        return matching_conversations[:limit]

    def _get_match_preview(self, content: str, query: str, context_chars: int = 100) -> str:
        """Get preview of content around the search match"""
        query_lower = query.lower()
        content_lower = content.lower()

        match_index = content_lower.find(query_lower)
        if match_index == -1:
            return content[:context_chars] + "..."

        start = max(0, match_index - context_chars // 2)
        end = min(len(content), match_index + len(query) + context_chars // 2)

        preview = content[start:end]
        if start > 0:
            preview = "..." + preview
        if end < len(content):
            preview = preview + "..."

        return preview

    def export_conversation(self, conversation_id: str, format: str = 'json') -> Dict[str, Any]:
        """Export conversation in specified format"""
        try:
            conversation = self.load_conversation(conversation_id)
            if not conversation:
                return {
                    'success': False,
                    'error': 'Conversation not found'
                }

            if format == 'json':
                return {
                    'success': True,
                    'data': conversation,
                    'filename': f"conversation_{conversation_id}.json"
                }

            elif format == 'txt':
                # Convert to readable text format
                text_content = f"Conversation: {conversation.get('title', 'Untitled')}\n"
                text_content += f"Created: {conversation.get('created_at', '')}\n"
                text_content += f"Updated: {conversation.get('updated_at', '')}\n"
                text_content += f"Messages: {len(conversation.get('messages', []))}\n\n"
                text_content += "=" * 50 + "\n\n"

                for message in conversation.get('messages', []):
                    role = message.get('role', 'unknown').upper()
                    timestamp = message.get('timestamp', '')
                    content = message.get('content', '')

                    text_content += f"[{timestamp}] {role}:\n{content}\n\n"
                    text_content += "-" * 30 + "\n\n"

                return {
                    'success': True,
                    'data': text_content,
                    'filename': f"conversation_{conversation_id}.txt"
                }

            elif format == 'md':
                # Convert to Markdown format
                md_content = f"# {conversation.get('title', 'Untitled Conversation')}\n\n"
                md_content += f"**Created:** {conversation.get('created_at', '')}\n"
                md_content += f"**Updated:** {conversation.get('updated_at', '')}\n"
                md_content += f"**Messages:** {len(conversation.get('messages', []))}\n\n"
                md_content += "---\n\n"

                for message in conversation.get('messages', []):
                    role = message.get('role', 'unknown')
                    timestamp = message.get('timestamp', '')
                    content = message.get('content', '')

                    if role == 'user':
                        md_content += f"## 👤 User ({timestamp})\n\n{content}\n\n"
                    elif role == 'assistant':
                        processing_time = message.get('processing_time', 0)
                        md_content += f"## 🤖 Assistant ({timestamp})\n"
                        if processing_time > 0:
                            md_content += f"*Processing time: {processing_time:.2f}s*\n\n"
                        md_content += f"{content}\n\n"

                    md_content += "---\n\n"

                return {
                    'success': True,
                    'data': md_content,
                    'filename': f"conversation_{conversation_id}.md"
                }

            else:
                return {
                    'success': False,
                    'error': f'Unsupported export format: {format}'
                }

        except Exception as e:
            logger.error(f"Error exporting conversation {conversation_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get analytics about conversations"""
        try:
            total_conversations = len(self.conversations_index)
            total_messages = 0
            total_processing_time = 0
            messages_per_day = {}
            avg_conversation_length = 0

            for conv_id in self.conversations_index.keys():
                conversation = self.load_conversation(conv_id)
                if conversation:
                    messages = conversation.get('messages', [])
                    total_messages += len(messages)

                    for message in messages:
                        # Count messages per day
                        timestamp = message.get('timestamp', '')
                        if timestamp:
                            try:
                                date = datetime.fromisoformat(timestamp).date().isoformat()
                                messages_per_day[date] = messages_per_day.get(date, 0) + 1
                            except:
                                pass

                        # Sum processing times
                        processing_time = message.get('processing_time', 0)
                        total_processing_time += processing_time

            if total_conversations > 0:
                avg_conversation_length = total_messages / total_conversations

            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'average_conversation_length': round(avg_conversation_length, 1),
                'total_processing_time': round(total_processing_time, 2),
                'average_processing_time': round(total_processing_time / max(1, total_messages), 2),
                'messages_per_day': messages_per_day,
                'most_active_day': max(messages_per_day.items(), key=lambda x: x[1]) if messages_per_day else None
            }

        except Exception as e:
            logger.error(f"Error getting conversation analytics: {e}")
            return {}

    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """Update conversation title"""
        try:
            conversation = self.load_conversation(conversation_id)
            if not conversation:
                return False

            conversation['title'] = new_title
            conversation['updated_at'] = datetime.now().isoformat()

            # Update index
            if conversation_id in self.conversations_index:
                self.conversations_index[conversation_id]['title'] = new_title
                self.conversations_index[conversation_id]['updated_at'] = conversation['updated_at']

            self._save_conversation(conversation_id, conversation)
            self._save_index()

            return True

        except Exception as e:
            logger.error(f"Error updating conversation title {conversation_id}: {e}")
            return False

    def bookmark_conversation(self, conversation_id: str, bookmarked: bool = True) -> bool:
        """Bookmark or unbookmark a conversation"""
        try:
            conversation = self.load_conversation(conversation_id)
            if not conversation:
                return False

            conversation['metadata']['bookmarked'] = bookmarked
            conversation['updated_at'] = datetime.now().isoformat()

            self._save_conversation(conversation_id, conversation)
            return True

        except Exception as e:
            logger.error(f"Error bookmarking conversation {conversation_id}: {e}")
            return False

    def get_bookmarked_conversations(self) -> List[Dict[str, Any]]:
        """Get all bookmarked conversations"""
        bookmarked = []

        for conv_id in self.conversations_index.keys():
            conversation = self.load_conversation(conv_id)
            if conversation and conversation.get('metadata', {}).get('bookmarked', False):
                bookmarked.append({
                    **self.conversations_index[conv_id],
                    'summary': self._get_conversation_summary(conv_id)
                })

        return sorted(bookmarked, key=lambda x: x.get('updated_at', ''), reverse=True)

    def cleanup_old_conversations(self, days: int = 30) -> Dict[str, Any]:
        """Clean up conversations older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            conversations_to_delete = []

            for conv_id, conv_info in self.conversations_index.items():
                try:
                    updated_at = datetime.fromisoformat(conv_info.get('updated_at', ''))
                    if updated_at < cutoff_date:
                        # Check if bookmarked (don't delete bookmarked conversations)
                        conversation = self.load_conversation(conv_id)
                        if not conversation or not conversation.get('metadata', {}).get('bookmarked', False):
                            conversations_to_delete.append(conv_id)
                except:
                    pass

            # Delete old conversations
            deleted_count = 0
            for conv_id in conversations_to_delete:
                if self.delete_conversation(conv_id):
                    deleted_count += 1

            return {
                'success': True,
                'deleted_count': deleted_count,
                'cutoff_date': cutoff_date.isoformat()
            }

        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_conversation_count(self) -> int:
        """Get total number of conversations"""
        return len(self.conversations_index)

    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most recent conversations"""
        conversations = list(self.conversations_index.values())
        conversations.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

        recent = []
        for conv_info in conversations[:limit]:
            recent.append({
                **conv_info,
                'summary': self._get_conversation_summary(conv_info['id'])
            })

        return recent