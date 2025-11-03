# Contributing to Genomic Sequence Detection

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, PyTorch version)

### Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Any implementation ideas you may have

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality

### Testing

Before submitting a pull request, please ensure:

```bash
# Run all tests
python tests/test_data.py
python tests/test_models.py
python tests/test_utils.py
```

### Documentation

- Update README.md if you change functionality
- Add docstrings to new functions and classes
- Update examples if necessary

## Development Setup

```bash
# Clone the repository
git clone https://github.com/arrdel/genomic_sequence_detection.git
cd genomic_sequence_detection

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue for any questions or concerns.

Thank you for contributing!
