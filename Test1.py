import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ollama
import threading
import PyPDF2
from pathlib import Path
import chardet
import json
from datetime import datetime
import os
import logging
from typing import Optional, Dict, Any
import markdown
from ttkwidgets.tooltips import Tooltip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ollama_ui.log'
)

class Config:
    def __init__(self):
        self.default_settings = {
            'model': 'deepseek-r1:7b',
            'theme': 'light',
            'window_size': '700x600',
            'font_size': 11,
            'auto_save': True,
            'history_limit': 100
        }
        self.settings_file = 'settings.json'
        self.current_settings = self.load()

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                return {**self.default_settings, **settings}
        except Exception as e:
            logging.warning(f"Failed to load settings: {e}")
            return self.default_settings.copy()

    def save(self) -> None:
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")

    def get(self, key: str) -> Any:
        return self.current_settings.get(key, self.default_settings.get(key))

    def set(self, key: str, value: Any) -> None:
        self.current_settings[key] = value
        self.save()

class StatusBar(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.label = ttk.Label(self, text="Ready")
        self.label.pack(side='left', padx=5)

    def update_status(self, text: str) -> None:
        self.label.configure(text=text)

class EnhancedOllamaUI:
    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.setup_ui()
        self.bind_shortcuts()
        self.setup_autosave()

    def setup_ui(self):
        # Window setup
        self.root.title("Enhanced Ollama Chat")
        self.root.geometry(self.config.get('window_size'))
        
        # Main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill='both', expand=True)

        # Menu bar
        self.create_menu()

        # Model selection with autocomplete
        self.create_model_selection()

        # Chat tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill='both', expand=True, pady=5)
        self.create_new_chat_tab()

        # Status bar
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(fill='x', side='bottom')

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Chat", command=self.create_new_chat_tab)
        file_menu.add_command(label="Save Chat", command=self.save_chat_history)
        file_menu.add_command(label="Export as PDF", command=self.export_as_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear History", command=self.clear_history)
        edit_menu.add_command(label="Copy Response", command=self.copy_response)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Dark Mode", command=self.toggle_theme)

    def create_model_selection(self):
        model_frame = ttk.Frame(self.main_container)
        model_frame.pack(fill='x', pady=5)

        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        self.model_var = tk.StringVar(value=self.config.get('model'))
        model_entry = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=self.get_available_models()
        )
        model_entry.pack(side='left', fill='x', expand=True, padx=5)

    def create_new_chat_tab(self):
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text=f"Chat {self.notebook.index('end')+1}")
        
        # Chat history
        history_frame = ttk.Frame(chat_frame)
        history_frame.pack(fill='both', expand=True, pady=5)

        self.chat_text = tk.Text(
            history_frame,
            wrap='word',
            font=('Helvetica', self.config.get('font_size')),
            height=15
        )
        self.chat_text.pack(fill='both', expand=True)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill='x', pady=5)

        self.input_text = tk.Text(
            input_frame,
            wrap='word',
            font=('Helvetica', self.config.get('font_size')),
            height=4
        )
        self.input_text.pack(fill='x', pady=5)

        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill='x')

        ttk.Button(
            button_frame,
            text="Send",
            command=self.send_message
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Upload File",
            command=self.upload_file
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Clear",
            command=lambda: self.input_text.delete('1.0', tk.END)
        ).pack(side='left', padx=5)

    def send_message(self):
        message = self.input_text.get('1.0', tk.END).strip()
        if not message:
            return

        self.chat_text.insert(tk.END, f"\nYou: {message}\n")
        self.input_text.delete('1.0', tk.END)
        
        # Start response in a thread
        threading.Thread(target=self.get_ollama_response, args=(message,)).start()

    def get_ollama_response(self, message: str):
        try:
            self.status_bar.update_status("Generating response...")
            response = ""
            
            for chunk in ollama.generate(
                self.model_var.get(),
                message,
                stream=True
            ):
                response += chunk['response']
                self.root.after(0, self.update_chat, response)

            self.root.after(0, self.status_bar.update_status, "Ready")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.show_error, error_msg)
            logging.error(error_msg)

    def update_chat(self, response: str):
        # Remove "<think>" tags before inserting the response
        cleaned_response = response.replace("<think>", "").replace("</think>", "").strip()

        # Delete the previous bot message if needed
        self.chat_text.delete('end-2c linestart', tk.END)

        # Insert the cleaned response
        self.chat_text.insert(tk.END, f"\nAssistant: {cleaned_response}\n")
        self.chat_text.see(tk.END)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return

        try:
            content = self.read_file(file_path)
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', f"Analyze this content:\n\n{content[:500]}...")
            self.status_bar.update_status(f"Loaded: {Path(file_path).name}")
        except Exception as e:
            self.show_error(f"Error loading file: {str(e)}")

    def read_file(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.read_pdf(file_path)
        else:
            return self.read_text(file_path)

    def read_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def read_text(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            raw = file.read()
            encoding = chardet.detect(raw)['encoding']
            
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()

    def show_error(self, message: str):
        messagebox.showerror("Error", message)
        self.status_bar.update_status("Error occurred")

    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")

        ttk.Label(settings_window, text="Font Size:").pack(pady=5)
        font_size = ttk.Scale(
            settings_window,
            from_=8,
            to=20,
            value=self.config.get('font_size'),
            command=lambda v: self.update_font_size(int(float(v)))
        )
        font_size.pack()

        ttk.Checkbutton(
            settings_window,
            text="Auto-save",
            variable=tk.BooleanVar(value=self.config.get('auto_save')),
            command=lambda: self.config.set('auto_save', not self.config.get('auto_save'))
        ).pack(pady=5)

    def update_font_size(self, size: int):
        self.config.set('font_size', size)
        self.chat_text.configure(font=('Helvetica', size))
        self.input_text.configure(font=('Helvetica', size))

    def toggle_theme(self):
        current_theme = self.config.get('theme')
        new_theme = 'dark' if current_theme == 'light' else 'light'
        self.config.set('theme', new_theme)
        self.apply_theme()

    def apply_theme(self):
        style = ttk.Style()
        if self.config.get('theme') == 'dark':
            # Apply dark theme colors
            style.configure('TFrame', background='#2d2d2d')
            style.configure('TLabel', background='#2d2d2d', foreground='white')
            self.chat_text.configure(bg='#3d3d3d', fg='white')
            self.input_text.configure(bg='#3d3d3d', fg='white')
        else:
            # Apply light theme colors
            style.configure('TFrame', background='white')
            style.configure('TLabel', background='white', foreground='black')
            self.chat_text.configure(bg='white', fg='black')
            self.input_text.configure(bg='white', fg='black')

    def bind_shortcuts(self):
        self.root.bind('<Control-Return>', lambda e: self.send_message())
        self.root.bind('<Control-n>', lambda e: self.create_new_chat_tab())
        self.root.bind('<Control-s>', lambda e: self.save_chat_history())

    def setup_autosave(self):
        if self.config.get('auto_save'):
            self.root.after(300000, self.auto_save)  # Save every 5 minutes

    def auto_save(self):
        if self.config.get('auto_save'):
            self.save_chat_history()
            self.root.after(300000, self.auto_save)

    def save_chat_history(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Chat History"
            )
            
            if not file_path:
                return
            
            history = {
                'model': self.model_var.get(),
                'content': self.chat_text.get('1.0', tk.END)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
                
            self.status_bar.update_status(f"Chat saved to {file_path}")
        except Exception as e:
            self.show_error(f"Error saving chat: {str(e)}")

    def export_as_pdf(self):
        # Implement PDF export functionality
        pass

    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the chat history?"):
            self.chat_text.delete('1.0', tk.END)

    def copy_response(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.chat_text.get('1.0', tk.END))

    def get_available_models(self) -> list:
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            if "deepseek-r1:1.5b" not in available_models:
                available_models.append("deepseek-r1:1.5b")
            return available_models
        except Exception as e:
            logging.error(f"Failed to get models: {e}")
            return [self.config.get('model'), "deepseek-r1:1.5b"]

def main():
    root = tk.Tk()
    app = EnhancedOllamaUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()