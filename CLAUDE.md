## Core Principles

## Development Workflow
1. Before making any changes, create and checkout a feature branch named `feature-[brief-description]`
2. Write comprehensive tests for all new functionality
3. Compile code and run all tests before committing
4. Write detailed commit messages explaining the changes and rationale
5. Commit all changes to the feature branch

## Code Quality Standards

Type Safety: Full type annotations throughout all Python code
Documentation: Google-style docstrings for all classes, methods, and functions
Error Handling: Comprehensive validation with informative error messages, graceful fallbacks
Architecture: Follow SOLID design principles for object-oriented code

## PyTorch Development Standards

Device Handling: Always implement MPS availability checks with CPU fallback
Memory Management: Explicit tensor device placement, efficient batch processing
Training Stability: Include gradient clipping, NaN detection, and loss monitoring
Checkpointing: Save full training state (model + optimizer + scheduler + metadata)

## Project Structure Conventions

Implementation: All source code in src/ directory
Models: Save trained models and checkpoints in models/ directory
Data: Load from data/preprocessed/ for processed datasets
Logging: Python logging to both console and file with structured messages

## Development Workflow

Configuration: Use dataclass-based configs for type safety and IDE support
Monitoring: Include progress bars (tqdm) and comprehensive metrics tracking
Validation: Built-in shape/dtype/device testing during execution
Extensibility: Design classes and interfaces for future model variations

## Dependencies Management

Core Stack: PyTorch, NumPy, scikit-learn, matplotlib, tqdm
Compatibility: Ensure Apple Silicon MPS optimization throughout
Versions: Use compatible versions for stable Apple Silicon performance

