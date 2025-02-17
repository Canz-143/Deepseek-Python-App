import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QProgressBar, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class LLMWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, question, model_name, temperature):
        super().__init__()
        self.question = question
        self.model_name = model_name
        self.temperature = temperature

    def run(self):
        try:
            self.progress.emit("Initializing LLM...")
            llm = Ollama(
                model=self.model_name,
                temperature=self.temperature
            )
            
            prompt = PromptTemplate(
                input_variables=["question"],
                template="Question: {question}\nDetailed Answer:"
            )
            
            self.progress.emit("Creating chain...")
            chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
            
            self.progress.emit("Getting response...")
            response = chain.invoke({"question": self.question})
            self.finished.emit(response["text"])
        except Exception as e:
            self.error.emit(str(e))

class LLMInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM Interface")
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "deepseek-r1:7b",
            "codellama:7b",
            "deepseek-r1:1.5b"
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        
        # Temperature control
        temp_label = QLabel("Temperature (0.0 - 1.0):")
        self.temp_selector = QComboBox()
        self.temp_selector.addItems(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        self.temp_selector.setCurrentText('0.7')
        model_layout.addWidget(temp_label)
        model_layout.addWidget(self.temp_selector)
        
        layout.addLayout(model_layout)
        
        # Input area
        input_label = QLabel("Question:")
        self.input_area = QTextEdit()
        self.input_area.setMaximumHeight(150)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        
        # Output area
        output_label = QLabel("Response:")
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        
        # Submit button
        self.submit_button = QPushButton("Ask LLM")
        self.submit_button.clicked.connect(self.process_question)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        
        # Add widgets to layout
        layout.addWidget(input_label)
        layout.addWidget(self.input_area)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(output_label)
        layout.addWidget(self.output_area)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTextEdit, QComboBox {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-height: 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
            QLabel {
                color: #333;
                font-weight: bold;
            }
        """)

    def process_question(self):
        question = self.input_area.toPlainText().strip()
        if not question:
            return
            
        # Disable input while processing
        self.submit_button.setEnabled(False)
        self.input_area.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.temp_selector.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        
        # Create and start worker thread
        self.worker = LLMWorker(
            question,
            self.model_selector.currentText(),
            float(self.temp_selector.currentText())
        )
        self.worker.finished.connect(self.handle_response)
        self.worker.error.connect(self.handle_error)
        self.worker.progress.connect(self.handle_progress)
        self.worker.start()

    def handle_response(self, response):
        self.output_area.setText(response)
        self.status_label.setText("Response complete")
        self._reset_ui()

    def handle_error(self, error_message):
        self.output_area.setText(f"Error: {error_message}")
        self.status_label.setText("Error occurred")
        self._reset_ui()

    def handle_progress(self, message):
        self.status_label.setText(message)

    def _reset_ui(self):
        self.submit_button.setEnabled(True)
        self.input_area.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.temp_selector.setEnabled(True)
        self.progress.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LLMInterface()
    window.show()
    sys.exit(app.exec())