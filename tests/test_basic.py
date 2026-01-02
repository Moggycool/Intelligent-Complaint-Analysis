"""
Basic tests for the Intelligent Complaint Analysis project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_directory_structure():
    """Test that required directories exist"""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        'data',
        'notebooks',
        'src',
        'vector_store',
        'tests',
        '.github',
        '.github/workflows'
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_required_files():
    """Test that required configuration files exist"""
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        'requirements.txt',
        'README.md',
        '.gitignore',
        'src/download_data.py',
        'src/eda_preprocessing.py',
        'src/create_vector_store.py',
        'notebooks/01_eda_preprocessing.ipynb',
        '.github/workflows/test.yml'
    ]
    
    for file_name in required_files:
        file_path = base_dir / file_name
        assert file_path.exists(), f"File {file_name} does not exist"
        assert file_path.is_file(), f"{file_name} is not a file"


def test_gitignore_content():
    """Test that .gitignore contains essential patterns"""
    base_dir = Path(__file__).parent.parent
    gitignore_path = base_dir / '.gitignore'
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    essential_patterns = [
        '__pycache__',
        'py[cod]',  # matches *.pyc, *.pyo, *.pyd
        'venv',
        '.env',
        'data/*.csv',
        'vector_store/*'
    ]
    
    for pattern in essential_patterns:
        assert pattern in content, f"Pattern {pattern} not found in .gitignore"


def test_readme_sections():
    """Test that README contains essential sections"""
    base_dir = Path(__file__).parent.parent
    readme_path = base_dir / 'README.md'
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    essential_sections = [
        'Project Overview',
        'Setup Instructions',
        'Usage',
        'Dependencies'
    ]
    
    for section in essential_sections:
        assert section in content, f"Section '{section}' not found in README"


if __name__ == "__main__":
    print("Running tests...")
    
    test_directory_structure()
    print("✓ Directory structure test passed")
    
    test_required_files()
    print("✓ Required files test passed")
    
    test_gitignore_content()
    print("✓ Gitignore content test passed")
    
    test_readme_sections()
    print("✓ README sections test passed")
    
    print("\n✅ All tests passed!")
