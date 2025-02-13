import tkinter as tk
from tkinter import ttk, filedialog
import ollama
import threading
import PyPDF2
from pathlib import Path
import chardet

# Optional imports with error handling
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

class MinimalistOllamaUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Chat")
        self.root.geometry("700x600")
        self.root.configure(bg='#ffffff')
        
        # Control flags
        self.stop_generation = False
        self.is_generating = False
        self.file_content = ""
        
        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', background='#ffffff')
        style.configure('TButton',
                       padding=10,
                       font=('Helvetica', 10))
        style.configure('Danger.TButton',
                       padding=10,
                       font=('Helvetica', 10),
                       background='#ff0000')
        style.configure('TLabel',
                       background='#ffffff',
                       font=('Helvetica', 10))
        
        # Main container
        self.main_container = ttk.Frame(root, padding="20 20 20 20")
        self.main_container.pack(fill='both', expand=True)
        
        # Model selection
        self.model_var = tk.StringVar(value="deepseek-r1:7b")
        model_entry = ttk.Entry(
            self.main_container,
            textvariable=self.model_var,
            font=('Helvetica', 11)
        )
        model_entry.pack(fill='x', pady=(0, 15))
        
        # File upload section
        self.file_frame = ttk.Frame(self.main_container)
        self.file_frame.pack(fill='x', pady=(0, 15))
        
        self.file_button = ttk.Button(
            self.file_frame,
            text="Upload File",
            command=self.upload_file
        )
        self.file_button.pack(side='left', padx=(0, 10))
        
        self.file_label = ttk.Label(
            self.file_frame,
            text="No file selected",
            background='#ffffff'
        )
        self.file_label.pack(side='left', fill='x', expand=True)
        
        # Prompt input area
        self.prompt_input = tk.Text(
            self.main_container,
            height=4,
            font=('Helvetica', 11),
            wrap='word',
            borderwidth=1,
            relief="solid",
            bg='#f8f8f8'
        )
        self.prompt_input.pack(fill='x', pady=(0, 10))
        
        # Button frame for submit and stop buttons
        self.button_frame = ttk.Frame(self.main_container)
        self.button_frame.pack(pady=(0, 15))
        
        # Submit button
        self.submit_button = ttk.Button(
            self.button_frame,
            text="Generate",
            command=self.submit_prompt
        )
        self.submit_button.pack(side='left', padx=(0, 5))
        
        # Stop button (hidden initially)
        self.stop_button = ttk.Button(
            self.button_frame,
            text="Stop",
            command=self.stop_generation_request,
            style='Danger.TButton'
        )
        
        # Response area
        self.response_output = tk.Text(
            self.main_container,
            font=('Helvetica', 11),
            wrap='word',
            borderwidth=1,
            relief="solid",
            bg='#f8f8f8'
        )
        self.response_output.pack(fill='both', expand=True)
        
        # Add placeholder text
        self.prompt_input.insert('1.0', 'Enter your prompt here...')
        self.prompt_input.bind('<FocusIn>', self.clear_placeholder)
        self.prompt_input.bind('<FocusOut>', self.restore_placeholder)

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def extract_text_from_docx(self, file_path):
        if not DOCX_SUPPORT:
            raise ImportError("python-docx package is not installed. Please install it using: pip install python-docx")
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def extract_text_from_txt(self, file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()

    def upload_file(self):
        filetypes = [
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
        ]
        
        if DOCX_SUPPORT:
            filetypes.insert(1, ("Word files", "*.docx"))
        
        filetypes.append(("All files", "*.*"))
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if file_path:
            try:
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension == '.pdf':
                    self.file_content = self.extract_text_from_pdf(file_path)
                elif file_extension == '.docx' and DOCX_SUPPORT:
                    self.file_content = self.extract_text_from_docx(file_path)
                else:
                    self.file_content = self.extract_text_from_txt(file_path)
                
                filename = Path(file_path).name
                self.file_label.configure(text=f"Loaded: {filename}")
                
                self.prompt_input.delete('1.0', tk.END)
                preview = self.file_content[:200] + "..." if len(self.file_content) > 200 else self.file_content
                self.prompt_input.insert('1.0', f"Analyze this text:\n\n{preview}")
                
            except Exception as e:
                self.file_label.configure(text=f"Error loading file: {str(e)}")
                self.file_content = ""

    def clear_placeholder(self, event):
        if self.prompt_input.get('1.0', 'end-1c') == 'Enter your prompt here...':
            self.prompt_input.delete('1.0', tk.END)
            self.prompt_input.configure(fg='black')

    def restore_placeholder(self, event):
        if not self.prompt_input.get('1.0', 'end-1c'):
            self.prompt_input.insert('1.0', 'Enter your prompt here...')
            self.prompt_input.configure(fg='gray')

    def stop_generation_request(self):
        self.stop_generation = True
        self.stop_button.configure(state='disabled', text="Stopping...")

    def submit_prompt(self):
        self.response_output.delete('1.0', tk.END)
        self.submit_button.pack_forget()  # Hide submit button
        self.stop_button.pack(side='left', padx=(0, 5))  # Show stop button
        self.stop_generation = False
        self.is_generating = True
        
        thread = threading.Thread(target=self.generate_response)
        thread.start()

    def generate_response(self):
        try:
            prompt = self.prompt_input.get('1.0', tk.END).strip()
            if prompt == 'Enter your prompt here...':
                prompt = ''
            
            if self.file_content:
                if not prompt.endswith(':'):
                    prompt += ':'
                prompt += f"\n\n{self.file_content}"
            
            model = self.model_var.get()
            
            for chunk in ollama.generate(model, prompt, stream=True):
                if self.stop_generation:
                    self.root.after(0, self.update_response, "\n\n[Generation stopped by user gwapo]")
                    break
                self.root.after(0, self.update_response, chunk["response"])
        except Exception as e:
            self.root.after(0, self.update_response, f"\nError: {str(e)}")
        finally:
            self.is_generating = False
            self.stop_generation = False
            self.root.after(0, self.reset_buttons)

    def reset_buttons(self):
        self.stop_button.pack_forget()  # Hide stop button
        self.submit_button.configure(state='normal')
        self.submit_button.pack(side='left', padx=(0, 5))  # Show submit button

    def update_response(self, text):
        self.response_output.insert(tk.END, text)
        self.response_output.see(tk.END)

def main():
    root = tk.Tk()
    app = MinimalistOllamaUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()