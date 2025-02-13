import tkinter as tk
from tkinter import ttk, filedialog
import ollama
import threading
import PyPDF2
from pathlib import Path
import chardet
import json
from datetime import datetime
import os

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
        
        # Theme and control flags
        self.is_dark_mode = False
        self.stop_generation = False
        self.is_generating = False
        self.file_content = ""
        self.chat_history = []
        
        # Color schemes
        self.colors = {
            'light': {
                'bg': '#ffffff',
                'fg': '#000000',
                'input_bg': '#f8f8f8',
                'button_bg': '#e0e0e0'
            },
            'dark': {
                'bg': '#2d2d2d',
                'fg': '#ffffff',
                'input_bg': '#3d3d3d',
                'button_bg': '#404040'
            }
        }
        
        # Configure styles
        self.style = ttk.Style()
        self.update_styles()
        
        # Main container
        self.main_container = ttk.Frame(root, padding="20 20 20 20")
        self.main_container.pack(fill='both', expand=True)
        
        # Top control frame
        self.control_frame = ttk.Frame(self.main_container)
        self.control_frame.pack(fill='x', pady=(0, 15))
        
        # Model selection
        self.model_var = tk.StringVar(value="deepseek-r1:7b")
        model_entry = ttk.Entry(
            self.control_frame,
            textvariable=self.model_var,
            font=('Helvetica', 11)
        )
        model_entry.pack(side='left', fill='x', expand=True)
        
        # Dark mode toggle
        self.dark_mode_btn = ttk.Button(
            self.control_frame,
            text="🌙 Dark",
            command=self.toggle_dark_mode,
            width=10
        )
        self.dark_mode_btn.pack(side='right', padx=(10, 0))
        
        # Save chat button
        self.save_chat_btn = ttk.Button(
            self.control_frame,
            text="💾 Save Chat",
            command=self.save_chat_history,
            width=12
        )
        self.save_chat_btn.pack(side='right', padx=(10, 0))
        
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
            text="No file selected"
        )
        self.file_label.pack(side='left', fill='x', expand=True)
        
        # Prompt input area
        self.prompt_input = tk.Text(
            self.main_container,
            height=4,
            font=('Helvetica', 11),
            wrap='word',
            borderwidth=1,
            relief="solid"
        )
        self.prompt_input.pack(fill='x', pady=(0, 10))
        
        # Button frame
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
            relief="solid"
        )
        self.response_output.pack(fill='both', expand=True)
        
        # Add placeholder text
        self.prompt_input.insert('1.0', 'Enter your prompt here...')
        self.prompt_input.bind('<FocusIn>', self.clear_placeholder)
        self.prompt_input.bind('<FocusOut>', self.restore_placeholder)
        
        # Initial theme application
        self.apply_theme()

    def update_styles(self):
        theme = 'dark' if self.is_dark_mode else 'light'
        colors = self.colors[theme]
        
        self.style.configure('TFrame', background=colors['bg'])
        self.style.configure('TButton', padding=10, font=('Helvetica', 10))
        self.style.configure('Danger.TButton', padding=10, font=('Helvetica', 10))
        self.style.configure('TLabel', background=colors['bg'], foreground=colors['fg'])

    def apply_theme(self):
        theme = 'dark' if self.is_dark_mode else 'light'
        colors = self.colors[theme]
        
        # Update root and main container
        self.root.configure(bg=colors['bg'])
        self.main_container.configure(style='TFrame')
        
        # Update text widgets
        for widget in [self.prompt_input, self.response_output]:
            widget.configure(
                bg=colors['input_bg'],
                fg=colors['fg'],
                insertbackground=colors['fg']
            )
        
        # Update dark mode button text
        self.dark_mode_btn.configure(text="☀️ Light" if self.is_dark_mode else "🌙 Dark")
        
        # Update all frames
        for frame in [self.control_frame, self.file_frame, self.button_frame]:
            frame.configure(style='TFrame')
        
        # Update labels
        self.file_label.configure(style='TLabel')

    def toggle_dark_mode(self):
        self.is_dark_mode = not self.is_dark_mode
        self.update_styles()
        self.apply_theme()

    def save_chat_history(self):
        if not self.chat_history and not self.response_output.get('1.0', tk.END).strip():
            return
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"chat_history_{timestamp}.json"
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=default_filename,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # Get current conversation if not in history
            current_prompt = self.prompt_input.get('1.0', tk.END).strip()
            current_response = self.response_output.get('1.0', tk.END).strip()
            
            if current_prompt and current_response:
                self.chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": current_prompt,
                    "response": current_response
                })
            
            # Save to file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model": self.model_var.get(),
                        "history": self.chat_history
                    }, f, indent=2, ensure_ascii=False)
                
                # Show success in file label temporarily
                original_text = self.file_label.cget("text")
                self.file_label.configure(text="Chat history saved successfully!")
                self.root.after(3000, lambda: self.file_label.configure(text=original_text))
            except Exception as e:
                self.file_label.configure(text=f"Error saving chat history: {str(e)}")

    def submit_prompt(self):
        prompt = self.prompt_input.get('1.0', tk.END).strip()
        if prompt == 'Enter your prompt here...':
            return
            
        self.response_output.delete('1.0', tk.END)
        self.submit_button.pack_forget()
        self.stop_button.pack(side='left', padx=(0, 5))
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
            full_response = ""
            
            for chunk in ollama.generate(model, prompt, stream=True):
                if self.stop_generation:
                    self.root.after(0, self.update_response, "\n\n[Generation stopped by user]")
                    break
                response_chunk = chunk["response"]
                full_response += response_chunk
                self.root.after(0, self.update_response, response_chunk)
            
            # Add to chat history
            if not self.stop_generation:
                self.chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "response": full_response
                })
            
        except Exception as e:
            self.root.after(0, self.update_response, f"\nError: {str(e)}")
        finally:
            self.is_generating = False
            self.stop_generation = False
            self.root.after(0, self.reset_buttons)

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