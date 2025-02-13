# Minimalist Deepseek UI

## Description
Minimalist Deepseek UI is a simple and intuitive chatbot interface built using Tkinter and powered by Ollama and Deepseek. It allows users to input prompts, generate responses, save chat history, and even analyze text files (TXT, PDF, DOCX) with the help of Deepseek's LLM.

## Features
- **Chat Interface**: Seamless interaction with Ollama AI
- **Advanced Model**: Uses the Deepseek model with 7B parameters
- **Model Flexibility**: Support for multiple models (default: deepseek-r1:7b)
- **File Analysis**: Upload and text extraction for TXT, PDF, and DOCX files
- **Customizable Theme**: Dark mode toggle for comfortable viewing
- **History Management**: Save chat history in JSON format
- **Control Options**: Stop response generation at any time
- **Clean Design**: Minimalist UI with responsive design

## Requirements

### Dependencies
Ensure you have the following Python packages installed:
```bash
pip install ollama PyPDF2 chardet
```

For DOCX file support, install:
```bash
pip install python-docx
```

### Ollama Installation
This application requires Ollama to run. Install it from:
https://ollama.com

**Important**: Ensure Ollama is running before launching the application.

## Installation & Usage

1. Clone this repository or download the script
2. Install the required dependencies
3. Run the application:
```bash
python Test1.py
```

## How to Use

1. **Chat Generation**: Enter a prompt in the text box and click "Generate" to receive a response
2. **File Analysis**: Upload a file to analyze its contents with AI
3. **Theme Customization**: Toggle between light and dark mode using the button
4. **History Management**: Save chat history for future reference
5. **Response Control**: Click "Stop" to interrupt response generation

## Notes

- If DOCX support is not installed, the script will still function but will not process DOCX files
- Ensure Ollama is running in the background before launching the script
