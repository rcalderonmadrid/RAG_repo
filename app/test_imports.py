#!/usr/bin/env python3
"""
Simple test script to verify that all core modules can be imported correctly.
Run this script before starting the application to check for import issues.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all core module imports"""
    print("🧪 Testing module imports...")

    errors = []

    # Test core modules
    modules_to_test = [
        ('config', 'Config'),
        ('utils', None),
        ('document_manager', 'DocumentManager'),
        ('conversation_manager', 'ConversationManager'),
        ('rag_engine', 'RAGEngine'),
    ]

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if class_name:
                getattr(module, class_name)
            print(f"✅ {module_name} - OK")
        except ImportError as e:
            error_msg = f"❌ {module_name} - Import Error: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
        except AttributeError as e:
            error_msg = f"❌ {module_name}.{class_name} - Class not found: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"❌ {module_name} - Unexpected error: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    # Test required external packages
    external_packages = [
        'streamlit',
        'langchain',
        'langchain_community',
        'langchain_ollama',
        'pandas',
        'plotly',
        'requests'
    ]

    print("\n🔍 Testing external dependencies...")

    for package in external_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            error_msg = f"❌ {package} - Not installed"
            print(error_msg)
            errors.append(error_msg)

    # Summary
    print(f"\n📊 Test Results:")
    if not errors:
        print("🎉 All imports successful! Ready to run the application.")
        return True
    else:
        print(f"⚠️ Found {len(errors)} issues:")
        for error in errors:
            print(f"  • {error}")
        print("\n💡 To fix:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Check that all module files are present")
        print("  3. Verify Python path includes current directory")
        return False

def check_file_structure():
    """Check that all required files exist"""
    print("\n📁 Checking file structure...")

    required_files = [
        'app.py',
        'config.py',
        'utils.py',
        'document_manager.py',
        'conversation_manager.py',
        'rag_engine.py',
        'requirements.txt',
        'README.md',
        'pages/chat.py',
        'pages/documents.py',
        'pages/conversations.py',
        'pages/analytics.py',
        'pages/settings.py'
    ]

    missing_files = []

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)

    if not missing_files:
        print("🎉 All required files present!")
        return True
    else:
        print(f"\n⚠️ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"  • {file_path}")
        return False

if __name__ == "__main__":
    print("🌊 RAG Intelligence Hub - Import Test")
    print("=" * 50)

    # Test file structure
    structure_ok = check_file_structure()

    # Test imports
    imports_ok = test_imports()

    print("=" * 50)

    if structure_ok and imports_ok:
        print("🚀 Ready to launch! Run: streamlit run app.py")
        sys.exit(0)
    else:
        print("🔧 Please fix the issues above before running the application.")
        sys.exit(1)