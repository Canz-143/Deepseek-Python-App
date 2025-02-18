import sys
import pandas as pd
import plotly.express as px
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QTextEdit, QPushButton, QLabel, QProgressBar, QComboBox,
                           QFileDialog, QHBoxLayout, QTabWidget, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tempfile
import os

class LLMWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, question, model_name, temperature, csv_context=None):
        super().__init__()
        self.question = question
        self.model_name = model_name
        self.temperature = temperature
        self.csv_context = csv_context

    def run(self):
        try:
            self.progress.emit("Initializing LLM...")
            llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature
            )
            
            # If we have CSV context, include it in the prompt
            if self.csv_context:
                template = """CSV Data Context:
{csv_context}

Question about the data: {question}

Provide a clear and detailed answer based on the CSV data above:"""
                prompt = PromptTemplate(
                    input_variables=["csv_context", "question"],
                    template=template
                )
                chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
                response = chain.invoke({
                    "csv_context": self.csv_context,
                    "question": self.question
                })
            else:
                prompt = PromptTemplate(
                    input_variables=["question"],
                    template="Question: {question}\nDetailed Answer:"
                )
                chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
                response = chain.invoke({"question": self.question})
            
            self.finished.emit(response["text"])
        except Exception as e:
            self.error.emit(str(e))

class LLMInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM Interface")
        self.setMinimumSize(1000, 800)  # Increased size for visualizations
        self.df = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Main tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        
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
        
        main_layout.addLayout(model_layout)
        
        # CSV File Upload Section
        file_section = QHBoxLayout()
        self.file_path_label = QLabel("No CSV file loaded")
        self.file_path_label.setStyleSheet("color: #666;")
        upload_button = QPushButton("Upload CSV")
        upload_button.clicked.connect(self.upload_csv)
        file_section.addWidget(self.file_path_label)
        file_section.addWidget(upload_button)
        main_layout.addLayout(file_section)
        
        # CSV Preview
        preview_label = QLabel("CSV Preview:")
        self.preview_area = QTextEdit()
        self.preview_area.setReadOnly(True)
        self.preview_area.setMaximumHeight(150)
        main_layout.addWidget(preview_label)
        main_layout.addWidget(self.preview_area)
        
        # Input area
        input_label = QLabel("Question:")
        self.input_area = QTextEdit()
        self.input_area.setMaximumHeight(100)
        main_layout.addWidget(input_label)
        main_layout.addWidget(self.input_area)
        
        # Submit button
        self.submit_button = QPushButton("Ask LLM")
        self.submit_button.clicked.connect(self.process_question)
        main_layout.addWidget(self.submit_button)
        
        # Status and Progress
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        main_layout.addWidget(self.status_label)
        
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        main_layout.addWidget(self.progress)
        
        # Output area
        output_label = QLabel("Response:")
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        main_layout.addWidget(output_label)
        main_layout.addWidget(self.output_area)
        
        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QVBoxLayout()
        
        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_label = QLabel("Plot Type:")
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Bar Chart", "Scatter Plot", "Line Plot", "Box Plot"])
        plot_type_layout.addWidget(plot_type_label)
        plot_type_layout.addWidget(self.plot_type)
        controls_layout.addLayout(plot_type_layout)
        
        # Column selection
        columns_layout = QHBoxLayout()
        
        x_label = QLabel("X Axis:")
        self.x_column = QComboBox()
        columns_layout.addWidget(x_label)
        columns_layout.addWidget(self.x_column)
        
        y_label = QLabel("Y Axis:")
        self.y_column = QComboBox()
        columns_layout.addWidget(y_label)
        columns_layout.addWidget(self.y_column)
        
        controls_layout.addLayout(columns_layout)
        
        # Create plot button
        self.plot_button = QPushButton("Create Plot")
        self.plot_button.clicked.connect(self.create_plot)
        controls_layout.addWidget(self.plot_button)
        
        controls_group.setLayout(controls_layout)
        viz_layout.addWidget(controls_group)
        
        # Plot display area
        self.web_view = QWebEngineView()
        viz_layout.addWidget(self.web_view)
        
        # Add tabs
        self.tabs.addTab(main_tab, "Chat")
        self.tabs.addTab(viz_tab, "Visualizations")
        
        layout.addWidget(self.tabs)
        
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

    def upload_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_name:
            try:
                self.df = pd.read_csv(file_name)
                self.file_path_label.setText(f"Loaded: {file_name}")
                
                # Update column selectors
                self.x_column.clear()
                self.y_column.clear()
                self.x_column.addItems(self.df.columns)
                self.y_column.addItems(self.df.columns)
                
                # Create preview
                preview = f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n"
                preview += f"Columns: {', '.join(self.df.columns)}\n\n"
                preview += "First few rows:\n"
                preview += self.df.head().to_string()
                
                self.preview_area.setText(preview)
                self.status_label.setText("CSV file loaded successfully")
            except Exception as e:
                self.status_label.setText(f"Error loading CSV: {str(e)}")
                self.df = None

    def create_plot(self):
        if self.df is None:
            self.status_label.setText("Please load a CSV file first")
            return
            
        try:
            x_col = self.x_column.currentText()
            y_col = self.y_column.currentText()
            plot_type = self.plot_type.currentText()
            
            if plot_type == "Bar Chart":
                fig = px.bar(self.df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif plot_type == "Scatter Plot":
                fig = px.scatter(self.df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif plot_type == "Line Plot":
                fig = px.line(self.df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            elif plot_type == "Box Plot":
                fig = px.box(self.df, x=x_col, y=y_col, title=f"{y_col} distribution by {x_col}")
            
            # Save to temporary file and display
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, 'temp_plot.html')
            fig.write_html(temp_path)
            
            self.web_view.setUrl(QUrl.fromLocalFile(temp_path))
            self.status_label.setText("Plot created successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error creating plot: {str(e)}")

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
        
        # Prepare CSV context if available
        csv_context = None
        if self.df is not None:
            csv_context = f"""The CSV file has {self.df.shape[0]} rows and {self.df.shape[1]} columns.
Column names: {', '.join(self.df.columns)}

First few rows of the data:
{self.df.head().to_string()}

Summary statistics:
{self.df.describe().to_string()}"""
        
        # Create and start worker thread
        self.worker = LLMWorker(
            question,
            self.model_selector.currentText(),
            float(self.temp_selector.currentText()),
            csv_context
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