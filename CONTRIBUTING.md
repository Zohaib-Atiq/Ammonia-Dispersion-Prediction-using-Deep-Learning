# Contributing to Ammonia Dispersion Prediction

First off, thank you for considering contributing to this project! It's people like you that make this research more accessible and valuable to the community.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if relevant**
- **Include your environment details** (Python version, OS, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed functionality**
- **Explain why this enhancement would be useful**
- **List some examples of how it would be used**

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Add or update tests if applicable
5. Update documentation if needed
6. Commit your changes:
   ```bash
   git commit -m "Add brief description of your changes"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Open a Pull Request

### Pull Request Guidelines

- **Keep changes focused**: One feature/fix per PR
- **Write clear commit messages**
- **Include comments in your code**
- **Update the README.md if needed**
- **Add tests for new features**
- **Ensure all tests pass**
- **Follow the existing code style**

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ammonia-dispersion-prediction.git
   cd ammonia-dispersion-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Comment complex logic

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions
- Update inline comments as needed
- Include examples for new features

### Testing

- Add tests for new features
- Ensure existing tests pass
- Aim for good test coverage
- Test edge cases

## Project Structure

```
.
├── ammonia_dispersion_ml_model.ipynb  # Main notebook
├── DrImran_paper4_machinelearning_v2.py  # Python script
├── unique_points.xlsx                 # Dataset
├── README.md                          # Main documentation
├── LICENSE                            # License file
├── requirements.txt                   # Dependencies
├── CONTRIBUTING.md                    # This file
└── .gitignore                         # Git ignore rules
```

## Questions?

Feel free to open an issue with your question or contact the maintainers directly.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing! 🎉
