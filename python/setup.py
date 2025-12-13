"""
Setup script for building qg Python wheel package
"""
import os
import sys
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
from pathlib import Path
import shutil

# Ensure we use g++ if available
if 'CC' not in os.environ:
    if shutil.which('g++'):
        os.environ['CC'] = 'g++'
        os.environ['CXX'] = 'g++'

# Get the project root directory (parent of python/)
# Use resolve() to get absolute path for include_dir, but use relative paths for source files
setup_file = Path(__file__).resolve()
project_root = setup_file.parent.parent.resolve()
include_dir = (project_root / "include").resolve()

# Check if include directory exists, if not, try to use environment variable or fallback
if not include_dir.exists():
    # If building from sdist in temp directory, try multiple strategies
    env_include = os.environ.get('RABITQ_INCLUDE_DIR')
    if env_include and Path(env_include).exists():
        include_dir = Path(env_include).resolve()
        print(f"[DEBUG] Using include dir from RABITQ_INCLUDE_DIR: {include_dir}")
    else:
        # Try relative path from setup.py (for sdist builds where include is in parent)
        # In sdist, the structure might be: qg-0.1.0/include/...
        relative_include = setup_file.parent.parent / "include"
        if relative_include.exists():
            include_dir = relative_include.resolve()
            print(f"[DEBUG] Using relative include dir (parent): {include_dir}")
        else:
            # Try same directory as setup.py (if include was copied there)
            same_dir_include = setup_file.parent / "include"
            if same_dir_include.exists():
                include_dir = same_dir_include.resolve()
                print(f"[DEBUG] Using same-dir include dir: {include_dir}")
            else:
                # Last resort: try hardcoded path (for development)
                hardcoded = Path("/home/lbl/code/graduationDesign/rabitq/testqg/external/RaBitQ-Library/include")
                if hardcoded.exists():
                    include_dir = hardcoded.resolve()
                    print(f"[DEBUG] Using hardcoded include dir: {include_dir}")
                else:
                    print(f"[WARNING] Include directory not found! Tried: {include_dir}")
                    print(f"[WARNING] Please set RABITQ_INCLUDE_DIR environment variable")

include_dir_str = str(include_dir)

# Custom build_ext to ensure we use g++ and include directories
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Ensure include directories are set correctly
        for ext in self.extensions:
            # Make sure include_dir_str is in the include_dirs (at the beginning)
            if include_dir_str not in ext.include_dirs:
                ext.include_dirs.insert(0, include_dir_str)
            
            # Print include directories for debugging
            print(f"\n[CustomBuildExt] Building extension {ext.name} with include directories:")
            for inc_dir in ext.include_dirs:
                inc_path = Path(inc_dir)
                exists = inc_path.exists() if inc_path.is_absolute() else False
                qg_hpp = (inc_path / 'rabitqlib/index/symqg/qg.hpp')
                qg_exists = qg_hpp.exists() if inc_path.is_absolute() else False
                print(f"  - {inc_dir}")
                print(f"    exists: {exists}, qg.hpp exists: {qg_exists}")
        
        # Override compiler if needed
        compiler_path = shutil.which('g++')
        if compiler_path and hasattr(self.compiler, 'compiler_so'):
            # Replace the compiler command with g++
            if self.compiler.compiler_so:
                self.compiler.compiler_so[0] = compiler_path
            if hasattr(self.compiler, 'compiler_cxx') and self.compiler.compiler_cxx:
                self.compiler.compiler_cxx[0] = compiler_path
        
        # Also set include_dirs on the compiler object directly
        if hasattr(self.compiler, 'include_dirs'):
            if include_dir_str not in self.compiler.include_dirs:
                self.compiler.include_dirs.insert(0, include_dir_str)
        
        # Print compiler command for debugging
        if hasattr(self.compiler, 'compiler_so') and self.compiler.compiler_so:
            print(f"[CustomBuildExt] Using compiler: {self.compiler.compiler_so[0]}")
        if hasattr(self.compiler, 'include_dirs'):
            print(f"[CustomBuildExt] Compiler include_dirs: {self.compiler.include_dirs}")
        
        super().build_extensions()
    
    def get_ext_filename(self, ext_name):
        # Override to print compilation details
        return super().get_ext_filename(ext_name)

# Debug: print paths (always print for now to debug)
print(f"[DEBUG] Setup file: {setup_file}")
print(f"[DEBUG] Project root: {project_root}")
print(f"[DEBUG] Include dir: {include_dir_str}")
print(f"[DEBUG] Include dir exists: {Path(include_dir_str).exists()}")
print(f"[DEBUG] qg.hpp exists: {(Path(include_dir_str) / 'rabitqlib/index/symqg/qg.hpp').exists()}")

# Compiler flags - explicitly add include directory
compile_args = [
    f"-I{include_dir_str}",  # Explicitly add include directory
    "-Wall", "-Ofast", "-Wextra", "-march=native", 
    "-fpic", "-fopenmp", "-ftree-vectorize", "-fexceptions",
    "-mavx2", "-mfma", "-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl",
    "-std=c++17"
]

# Linker flags
link_args = ["-fopenmp", "-lrt"]

# Define the extension module
# Source files must use relative paths from setup.py directory
ext_modules = [
    Pybind11Extension(
        "qg._qg_core",
        [
            "qg_bindings.cpp",  # Relative path from python/ directory
        ],
        include_dirs=[
            include_dir_str,  # Absolute path for RaBitQ headers
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="qg",
    version="0.1.0",
    author="RaBitQ",
    description="Python bindings for RaBitQ QuantizedGraph",
    long_description="",
    ext_modules=ext_modules,
    packages=["qg"],
    package_dir={"qg": "qg"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)

