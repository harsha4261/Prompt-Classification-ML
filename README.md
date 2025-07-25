This README provides a comprehensive overview of the `Prompt-Classification-ML` project.

---

# Prompt-Classification-ML

[![GitHub Repository Size](https://img.shields.io/github/repo-size/harsha4261/Prompt-Classification-ML)](https://github.com/harsha4261/Prompt-Classification-ML)
[![GitHub last commit](https://img.shields.io/github/last-commit/harsha4261/Prompt-Classification-ML)](https://github.com/harsha4261/Prompt-Classification-ML/commits/main)
[![GitHub Issues](https://img.shields.io/github/issues/harsha4261/Prompt-Classification-ML)](https://github.com/harsha4261/Prompt-Classification-ML/issues)
[![GitHub Stars](https://img.shields.io/github/stars/harsha4261/Prompt-Classification-ML?style=social)](https://github.com/harsha4261/Prompt-Classification-ML/stargazers)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Screenshots/Examples](#screenshots-examples)

---

## Description

This repository hosts a Machine Learning project dedicated to the classification of textual prompts. The project aims to accurately categorize various prompts using advanced ML techniques, including hierarchical modeling. It covers the entire machine learning pipeline, from data collection and preprocessing to model training, evaluation, and deployment of a simple web application for real-time inference.

The core of the project involves:
*   **Data Preparation**: Handling and expanding prompt datasets.
*   **Model Building**: Experimenting with different machine learning algorithms and architectures, notably exploring hierarchical classification approaches.
*   **Model Persistence**: Saving trained models and preprocessors for efficient use.
*   **Application Development**: Providing a user-friendly interface to classify new prompts.

The repository includes Jupyter notebooks for experimental development, Python scripts for core functionalities, and documentation files detailing the project's methodology and results.

---

## Features

*   **Prompt Classification**: Core functionality to classify input text prompts into predefined categories.
*   **Hierarchical Model Building**: Implementation and exploration of hierarchical machine learning models for improved classification accuracy and structure.
*   **Data Preprocessing**: Robust techniques for cleaning, tokenizing, and vectorizing textual data (e.g., using TF-IDF).
*   **Model Persistence**: Pre-trained vectorizers and classification models are saved (`.pkl` files) for quick loading and inference.
*   **Data Augmentation/Generation**: Includes datasets that appear to be generated or expanded (`expanded_prompts.csv`, `generated_prompts_01.csv`) to enhance training diversity.
*   **Interactive Web Application**: A lightweight web application (`app.py`) for users to input prompts and receive real-time classification results.
*   **Jupyter Notebooks**: Comprehensive notebooks documenting the data exploration, model training, and evaluation processes.
*   **Project Documentation**: Detailed reports (`.docx`, `.pdf`) outlining the project's approach, findings, and technical aspects.

---

## Project Structure

The repository is organized to clearly separate data, code, and documentation:

```
Prompt-Classification-ML/
├── .ipynb_checkpoints/          # Jupyter Notebook checkpoints (temporary files)
├── data/                        # Datasets used for training and testing
│   └── expanded_prompts.csv
├── docx/                        # Project reports and documentation
│   ├── 23BQ1A4261_ML_Report.docx.docx
│   └── 23BQ1A4261_prompt-classification report.pdf
├── 23BQ1A4261_Prompt-Classification.ipynb # Main Jupyter notebook for project development
├── 23BQ1A4261_Prompt-Classification.py    # Python script version of the main project
├── app.py                       # Python script for the web application (likely Streamlit/Flask)
├── cluster_encoder.pkl          # Pickled object, possibly a trained cluster encoder
├── vectorizer.pkl               # Pickled object, typically a TF-IDF vectorizer or similar
├── README.md                    # This README file
└── __pycache__/                 # Python bytecode cache
```

**Key Files and Directories:**
*   `data/`: Contains the primary dataset, `expanded_prompts.csv`, which is central to model training.
*   `docx/`: Holds the official project reports, offering deep insights into the methodologies and results.
*   `23BQ1A4261_Prompt-Classification.ipynb`: The main Jupyter notebook where data loading, preprocessing, model training, and evaluation are performed.
*   `app.py`: The entry point for the web application, allowing users to classify prompts via a user interface.
*   `vectorizer.pkl`: A serialized object (e.g., `TfidfVectorizer`) used to transform text into numerical features, crucial for consistency between training and inference.
*   `cluster_encoder.pkl`: A serialized encoder, potentially related to clustering or embedding for the hierarchical model.

---

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harsha4261/Prompt-Classification-ML.git
    cd Prompt-Classification-ML
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    This project likely uses standard Python libraries for machine learning and NLP. Since a `requirements.txt` is not explicitly provided, you might need to install them manually.
    ```bash
    pip install pandas numpy scikit-learn nltk
    # If app.py is a Streamlit app
    pip install streamlit
    # If app.py is a Flask app
    # pip install flask
    ```
    *Note: You might need to install NLTK data for specific tokenizers or resources if used in the notebooks.*
    ```python
    import nltk
    nltk.download('punkt') # Example, install as needed
    ```

---

## Usage

### 1. Explore and Train Models (Jupyter Notebook)

The main development and experimentation happen in the Jupyter notebooks.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  Open `23BQ1A4261_Prompt-Classification.ipynb` in your browser.
3.  Execute the cells sequentially to:
    *   Load and preprocess data from `data/expanded_prompts.csv`.
    *   Train the text vectorizer (`vectorizer.pkl`).
    *   Train various classification models, including the hierarchical model.
    *   Evaluate model performance.
    *   Save trained models and `vectorizer.pkl`, `cluster_encoder.pkl`.
    *   Other notebooks like `enhanced_model_building-checkpoint.ipynb` and `hierarchical_model_building-checkpoint.ipynb` (from `.ipynb_checkpoints`) can provide more details on specific model development phases.

### 2. Run the Web Application

The `app.py` script provides a user interface for classifying prompts.

1.  Ensure you have run the Jupyter notebook at least once to generate the `vectorizer.pkl` and `cluster_encoder.pkl` (and any other model `.pkl` files) as the `app.py` relies on these pre-trained components.
2.  **Run the application:**
    If `app.py` is a Streamlit application:
    ```bash
    streamlit run app.py
    ```
    If `app.py` is a Flask application (common, but requires `FLASK_APP=app.py`):
    ```bash
    export FLASK_APP=app.py # For macOS/Linux
    # set FLASK_APP=app.py # For Windows CMD
    flask run
    ```
3.  Open your web browser and navigate to the address provided by the command line (e.g., `http://localhost:8501` for Streamlit or `http://127.0.0.1:5000` for Flask).
4.  Enter a prompt in the input field and observe the classification result.

---

## License

No explicit license file was found in the repository. Therefore, the project is currently **unlicensed**. Users should contact the repository owner for permissions regarding use, distribution, or modification.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** (`git checkout -b feature/YourFeatureName` or `bugfix/FixDescription`).
3.  **Make your changes**.
4.  **Commit your changes** (`git commit -m 'Add new feature'`).
5.  **Push to the branch** (`git push origin feature/YourFeatureName`).
6.  **Open a Pull Request** to the `main` branch of the original repository.

Please ensure your code adheres to good practices and includes comments where necessary.

---

## Acknowledgements

*   The creator of this repository, [Harsha](https://github.com/harsha4261), for developing this project.
*   The open-source community for providing the tools and libraries (e.g., `scikit-learn`, `pandas`, `streamlit`) that make projects like this possible.

---
