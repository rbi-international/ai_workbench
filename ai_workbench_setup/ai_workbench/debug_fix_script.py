#!/usr/bin/env python3
"""
Quick fix for import errors - adds clean_response_text import to all task files
"""

import os
import re

def fix_imports_in_file(file_path, module_name):
    """Fix imports in a specific file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if clean_response_text is already imported
        if 'clean_response_text' in content:
            print(f"✅ {file_path} already has clean_response_text import")
            return False
        
        # Find the helpers import line and add clean_response_text
        old_import = "from utils.helpers import validate_text, validate_model_params"
        new_import = "from utils.helpers import validate_text, validate_model_params, clean_response_text"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print(f"✅ Added import to {file_path}")
        else:
            # Alternative import patterns
            patterns = [
                (r"from utils\.helpers import ([^,\n]+)", r"from utils.helpers import \1, clean_response_text"),
                (r"from utils import helpers", "from utils import helpers\nfrom utils.helpers import clean_response_text")
            ]
            
            for old_pattern, new_pattern in patterns:
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    print(f"✅ Added import to {file_path} (pattern match)")
                    break
            else:
                # If no import found, add at top
                content = "from utils.helpers import clean_response_text\n" + content
                print(f"✅ Added import to {file_path} (at top)")
        
        # Now fix the usage - look for .strip() calls and replace with clean_response_text
        patterns_to_fix = [
            (r'(\w+)\.strip\(\)', r'clean_response_text(\1.strip())'),
            (r'"output": response\.strip\(\),', '"output": clean_response_text(response.strip()),'),
            (r'"output": summary\.strip\(\),', '"output": clean_response_text(summary.strip()),'),
            (r'"output": translation\.strip\(\),', '"output": clean_response_text(translation.strip()),'),
        ]
        
        for old_pattern, new_pattern in patterns_to_fix:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_pattern, content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix import errors in all task files"""
    print("🔧 Quick Import Fix for clean_response_text")
    print("=" * 50)
    
    # Files that need the import fix
    files_to_fix = [
        "src/tasks/chatter.py",
        "src/tasks/summarizer.py", 
        "src/tasks/translator.py",
        "src/models/openai_model.py",
        "src/models/llama_model.py"
    ]
    
    fixes_applied = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"\n🔄 Fixing {file_path}...")
            if fix_imports_in_file(file_path, os.path.basename(file_path)):
                fixes_applied += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Also check if helpers.py has the function
    print(f"\n📋 Checking utils/helpers.py...")
    if os.path.exists("utils/helpers.py"):
        with open("utils/helpers.py", 'r') as f:
            helpers_content = f.read()
        
        if "def clean_response_text" in helpers_content:
            print("✅ clean_response_text function exists in helpers.py")
        else:
            print("❌ clean_response_text function NOT found in helpers.py")
            print("   Run the debug_fix_script.py first!")
    
    print(f"\n🎉 Import fix complete! Applied fixes to {fixes_applied} files.")
    print("\n🚀 Next steps:")
    print("1. Restart your API server: python main.py")
    print("2. Test chat functionality")
    print("3. If still errors, check the console for specific missing imports")
    
    # Clear cache to avoid cached responses
    print("\n🗑️  Clearing cache...")
    cache_dir = "data/cache"
    if os.path.exists(cache_dir):
        try:
            import shutil
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            print("✅ Cache cleared - fresh responses will be generated")
        except Exception as e:
            print(f"⚠️  Could not clear cache: {e}")

if __name__ == "__main__":
    main()