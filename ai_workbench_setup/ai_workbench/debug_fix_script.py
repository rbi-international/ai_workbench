#!/usr/bin/env python3
"""
AI Workbench Cleanup Script
Removes all voice-related and unwanted files
"""

import os
import shutil
import glob
from pathlib import Path

def remove_file(file_path):
    """Safely remove a file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed file: {file_path}")
            return True
        else:
            print(f"⚠️  File not found: {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error removing {file_path}: {e}")
        return False

def remove_directory(dir_path):
    """Safely remove a directory and all its contents"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Removed directory: {dir_path}")
            return True
        else:
            print(f"⚠️  Directory not found: {dir_path}")
            return False
    except Exception as e:
        print(f"❌ Error removing directory {dir_path}: {e}")
        return False

def cleanup_voice_files():
    """Remove all voice-related files"""
    print("🔊 Removing voice-related files...")
    
    voice_files = [
        "ai_workbench_setup/ai_workbench/src/voice.py",
        "ai_workbench_setup/ai_workbench/src/voice_interface.py",
        "src/voice.py",
        "src/voice_interface.py"
    ]
    
    for file_path in voice_files:
        remove_file(file_path)

def cleanup_voice_directories():
    """Remove voice-related directories"""
    print("📁 Removing voice-related directories...")
    
    voice_dirs = [
        "data/voice_output",
        "ai_workbench_setup/ai_workbench/data/voice_output"
    ]
    
    for dir_path in voice_dirs:
        remove_directory(dir_path)

def cleanup_cache_files():
    """Remove cache and temporary files"""
    print("🗑️  Removing cache and temporary files...")
    
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo", 
        "**/*.pyd",
        "**/.pytest_cache",
        "**/node_modules",
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.log",
        "**/logs/*.log"
    ]
    
    for pattern in cache_patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                remove_file(path)
            elif os.path.isdir(path):
                remove_directory(path)

def cleanup_build_files():
    """Remove build and distribution files"""
    print("🏗️  Removing build files...")
    
    build_dirs = [
        "build",
        "dist",
        "*.egg-info",
        ".eggs",
        "ai_workbench_setup/ai_workbench/build",
        "ai_workbench_setup/ai_workbench/dist",
        "ai_workbench_setup/ai_workbench/*.egg-info"
    ]
    
    for pattern in build_dirs:
        for path in glob.glob(pattern, recursive=True):
            if os.path.exists(path):
                remove_directory(path)

def cleanup_ide_files():
    """Remove IDE and editor files"""
    print("💻 Removing IDE files...")
    
    ide_patterns = [
        "**/.vscode",
        "**/.idea",
        "**/*.swp",
        "**/*.swo",
        "**/*~",
        "**/.project",
        "**/.pydevproject"
    ]
    
    for pattern in ide_patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                remove_file(path)
            elif os.path.isdir(path):
                remove_directory(path)

def cleanup_unwanted_directories():
    """Remove entire unwanted directory structures"""
    print("📂 Removing unwanted directory structures...")
    
    unwanted_dirs = [
        "ai_workbench_setup",  # Remove the entire setup directory if you want
        ".git",  # Remove if you want to clean git history
        # Add other unwanted directories here
    ]
    
    # Ask user confirmation for major directories
    for dir_path in unwanted_dirs:
        if os.path.exists(dir_path):
            if dir_path in [".git", "ai_workbench_setup"]:
                response = input(f"⚠️  Remove {dir_path}? This will delete all contents. (y/N): ")
                if response.lower() in ['y', 'yes']:
                    remove_directory(dir_path)
                else:
                    print(f"⏭️  Skipped: {dir_path}")
            else:
                remove_directory(dir_path)

def cleanup_audio_files():
    """Remove any audio files that might be lingering"""
    print("🎵 Removing audio files...")
    
    audio_patterns = [
        "**/*.mp3",
        "**/*.wav", 
        "**/*.ogg",
        "**/*.m4a",
        "**/*.flac",
        "data/**/*.mp3",
        "data/**/*.wav"
    ]
    
    for pattern in audio_patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                # Ask before deleting audio files in case they're important
                response = input(f"⚠️  Remove audio file {path}? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    remove_file(path)
                else:
                    print(f"⏭️  Skipped: {path}")

def cleanup_test_files():
    """Remove test files and test data"""
    print("🧪 Removing test files...")
    
    test_patterns = [
        "**/test_*.py",
        "**/tests/**",
        "**/*_test.py",
        "**/pytest.ini",
        "**/tox.ini",
        "**/.coverage",
        "**/htmlcov",
        "**/coverage.xml"
    ]
    
    for pattern in test_patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                remove_file(path)
            elif os.path.isdir(path):
                remove_directory(path)

def cleanup_documentation():
    """Remove unwanted documentation files"""
    print("📚 Cleaning up documentation...")
    
    doc_files = [
        "README_old.md",
        "CHANGELOG.md",
        "TODO.md",
        "docs/**/*.md",
        "**/*.rst"
    ]
    
    for pattern in doc_files:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                # Ask before deleting docs
                response = input(f"⚠️  Remove documentation file {path}? (y/N): ")
                if response.lower() in ['y', 'yes']:
                    remove_file(path)

def create_gitignore():
    """Create/update .gitignore file"""
    print("📝 Creating/updating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Cache
.pytest_cache/
.coverage
htmlcov/

# Data
data/cache/
data/voice_output/
*.db
chroma_db/

# Model files
models/
*.pt
*.pth
*.bin

# Environment variables
.env.local
.env.production

# Temporary files
tmp/
temp/
*.tmp
"""
    
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created/updated .gitignore")
    except Exception as e:
        print(f"❌ Error creating .gitignore: {e}")

def main():
    """Main cleanup function"""
    print("🧹 AI Workbench Cleanup Script")
    print("=" * 40)
    print("This script will remove:")
    print("- Voice-related files and directories")
    print("- Cache files (__pycache__, *.pyc, etc.)")
    print("- Build files (build/, dist/, *.egg-info)")
    print("- IDE files (.vscode/, .idea/, etc.)")
    print("- Temporary and log files")
    print("- Test files (optional)")
    print("- Audio files (with confirmation)")
    print("- Documentation files (with confirmation)")
    print()
    
    response = input("🤔 Continue with cleanup? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Cleanup cancelled")
        return
    
    print("\n🚀 Starting cleanup...")
    
    # Core cleanup (always safe)
    cleanup_voice_files()
    cleanup_voice_directories()
    cleanup_cache_files()
    cleanup_build_files()
    cleanup_ide_files()
    
    # Optional cleanup (with user confirmation)
    print("\n📋 Optional cleanup steps:")
    
    if input("Remove test files? (y/N): ").lower() in ['y', 'yes']:
        cleanup_test_files()
    
    if input("Remove audio files? (y/N): ").lower() in ['y', 'yes']:
        cleanup_audio_files()
    
    if input("Clean documentation files? (y/N): ").lower() in ['y', 'yes']:
        cleanup_documentation()
    
    if input("Remove unwanted directories (including setup folder)? (y/N): ").lower() in ['y', 'yes']:
        cleanup_unwanted_directories()
    
    # Create .gitignore
    if input("Create/update .gitignore? (y/N): ").lower() in ['y', 'yes']:
        create_gitignore()
    
    print("\n✅ Cleanup completed!")
    print("🎯 Your AI Workbench is now clean and optimized")
    print("\n📁 Recommended next steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Update your .env file with API keys")
    print("3. Run: python startup.py")

if __name__ == "__main__":
    main()