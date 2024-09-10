# Chat with Data Web App

## Overview

The **Chat with Data Web App** is a Streamlit application that allows users to interact with different types of data through a conversational interface. Users can upload PDFs, text files, CSV files, SQL databases, or provide website URLs to query and retrieve information. The application uses language models and vector stores to process and answer user queries based on the provided data.

## Features

- **Chat with PDFs**: Upload PDF files and ask questions about their content.
- **Chat with Text Files**: Upload text files (.txt) and interact with their content.
- **Chat with CSV Files**: Upload CSV file and query the data within.
- **Chat with SQL Databases**: Upload SQLite database and run SQL queries through a chatbot interface.
- **Chat with Websites**: Provide website URLs to extract and query information from web pages.

## Installation

To get started, clone the repository and install the required packages. Ensure you have Python 3.7+ installed.

### Clone the Repository

```bash
git clone https://github.com/kaustubh-tr/chat-with-data.git
cd chat-with-data
```

### Install Dependencies

Create a virtual environment (optional but recommended) and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

To start the application, run:

```bash
streamlit run main_app.py
```

Open your browser and navigate to `http://localhost:8501` to interact with the application.

### Features

1. **Chat with PDFs**
   - Upload PDF files through the sidebar.
   - Ask questions about the content of the PDFs.

2. **Chat with Text Files**
   - Upload text files (.txt) through the sidebar.
   - Ask questions about the content of the text files.

3. **Chat with CSV Files**
   - Upload a CSV file through the sidebar.
   - Query data from the CSV files in natural language.

4. **Chat with SQL Databases**
   - Upload SQLite database file (.db) through the sidebar.
   - Run SQL queries through the chatbot in natural language.

5. **Chat with Websites**
   - Enter website URLs through the sidebar.
   - Ask questions about the content of the provided websites.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Open a pull request.
