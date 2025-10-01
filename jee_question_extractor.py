"""
JEE Question Paper Extractor with GUI
Extracts questions from JEE PDF papers to Excel with high accuracy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from pix2text import Pix2Text
import cv2
from datetime import datetime
import traceback


class QuestionExtractor:
    """Core extraction logic with error reduction techniques"""
    
    def __init__(self, progress_callback=None, log_callback=None):
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.p2t = None
        self.initialize_ocr()
        
    def initialize_ocr(self):
        """Initialize Pix2Text with optimized settings"""
        try:
            self.log("Initializing OCR engine...")
            self.p2t = Pix2Text(
                analyzer_config=dict(
                    model_name='mfd',
                    model_backend='onnx',
                ),
                formula_config=dict(
                    model_name='mfr',
                    model_backend='onnx',
                )
            )
            self.log("OCR engine initialized successfully")
        except Exception as e:
            self.log(f"Error initializing OCR: {str(e)}")
            # Fallback to basic configuration
            self.p2t = Pix2Text()
    
    def log(self, message):
        """Log message through callback"""
        if self.log_callback:
            self.log_callback(message)
        print(message)
    
    def update_progress(self, value, message=""):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback(value, message)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlDenoising(gray, h=10)
        
        # Apply adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Convert PDF pages to high-resolution images"""
        images = []
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc):
                self.update_progress(
                    int((page_num / total_pages) * 30),
                    f"Converting page {page_num + 1}/{total_pages}"
                )
                
                # Render page at high DPI
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Preprocess image
                processed = self.preprocess_image(img_array)
                images.append(processed)
                
            doc.close()
            self.log(f"Converted {total_pages} pages to images")
            
        except Exception as e:
            self.log(f"Error converting PDF: {str(e)}")
            raise
            
        return images
    
    def extract_text_from_image(self, image: np.ndarray) -> str:
        """Extract text from image using Pix2Text"""
        try:
            result = self.p2t.recognize(image, resized_shape=1920)
            
            # Combine text and latex formulas
            text_parts = []
            for item in result:
                if item['type'] == 'text':
                    text_parts.append(item['text'])
                elif item['type'] == 'formula':
                    # Preserve LaTeX formulas
                    text_parts.append(f"$$${item['text']}$$$")
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            self.log(f"OCR error: {str(e)}")
            return ""
    
    def identify_question_patterns(self, text: str) -> List[Dict]:
        """Identify and extract questions using multiple patterns"""
        questions = []
        
        # Multiple regex patterns for different question formats
        patterns = [
            # Pattern 1: Numbered questions (1., 2., etc.)
            r'(?:^|\n)\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)',
            # Pattern 2: Questions with Q prefix
            r'(?:^|\n)\s*Q\.?\s*(\d+)\s*[:.]?\s*(.+?)(?=\n\s*Q\.|$)',
            # Pattern 3: Questions in brackets
            r'(?:^|\n)\s*\[(\d+)\]\s*(.+?)(?=\n\s*\[\d+\]|$)',
            # Pattern 4: Questions with parentheses
            r'(?:^|\n)\s*\((\d+)\)\s*(.+?)(?=\n\s*\(\d+\)|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                q_num = match.group(1)
                q_text = match.group(2).strip()
                
                # Extract options if present
                options = self.extract_options(q_text)
                
                # Clean question text
                q_text_clean = self.clean_question_text(q_text)
                
                if q_text_clean and len(q_text_clean) > 10:  # Filter out too short texts
                    questions.append({
                        'number': int(q_num),
                        'question': q_text_clean,
                        'options': options,
                        'raw_text': q_text
                    })
        
        # Remove duplicates based on question number
        seen = set()
        unique_questions = []
        for q in sorted(questions, key=lambda x: x['number']):
            if q['number'] not in seen:
                seen.add(q['number'])
                unique_questions.append(q)
        
        return unique_questions
    
    def extract_options(self, text: str) -> Dict[str, str]:
        """Extract multiple choice options from question text"""
        options = {}
        
        # Common option patterns
        option_patterns = [
            r'(?:^|\n)\s*\(([A-D])\)\s*(.+?)(?=\n\s*\([A-D]\)|$)',
            r'(?:^|\n)\s*([A-D])\.\s*(.+?)(?=\n\s*[A-D]\.|$)',
            r'(?:^|\n)\s*([a-d])\)\s*(.+?)(?=\n\s*[a-d]\)|$)',
        ]
        
        for pattern in option_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                opt_letter = match.group(1).upper()
                opt_text = match.group(2).strip()
                options[opt_letter] = self.clean_option_text(opt_text)
        
        return options
    
    def clean_question_text(self, text: str) -> str:
        """Clean and format question text"""
        # Remove option text from question
        text = re.sub(r'\n\s*\([A-Da-d]\).*', '', text)
        text = re.sub(r'\n\s*[A-Da-d]\..*', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Fix common OCR errors
        replacements = {
            ' ,': ',',
            ' .': '.',
            ' ;': ';',
            ' :': ':',
            '( ': '(',
            ' )': ')',
            '[ ': '[',
            ' ]': ']',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_option_text(self, text: str) -> str:
        """Clean option text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove trailing punctuation if it's just a period
        if text.endswith('.') and text.count('.') == 1:
            text = text[:-1]
        
        return text
    
    def post_process_questions(self, questions: List[Dict]) -> List[Dict]:
        """Post-process questions to improve quality"""
        processed = []
        
        for q in questions:
            # Check for mathematical formulas and preserve them
            q['has_formula'] = '$$$' in q['question']
            
            # Estimate question type
            q['type'] = self.identify_question_type(q['question'])
            
            # Add metadata
            q['char_count'] = len(q['question'])
            q['word_count'] = len(q['question'].split())
            
            processed.append(q)
        
        return processed
    
    def identify_question_type(self, text: str) -> str:
        """Identify the type of question"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['calculate', 'find', 'determine', 'compute']):
            return 'Numerical'
        elif any(word in text_lower for word in ['which', 'what', 'choose', 'select']):
            if 'correct' in text_lower or 'true' in text_lower:
                return 'MCQ'
            return 'Descriptive'
        elif 'match' in text_lower:
            return 'Match'
        elif any(word in text_lower for word in ['true', 'false']):
            return 'True/False'
        else:
            return 'Other'
    
    def extract_from_pdf(self, pdf_path: str, output_path: str) -> Tuple[bool, str]:
        """Main extraction function"""
        try:
            self.log(f"Starting extraction from: {pdf_path}")
            
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path, dpi=300)
            
            all_questions = []
            total_images = len(images)
            
            # Process each page
            for i, image in enumerate(images):
                self.update_progress(
                    30 + int((i / total_images) * 50),
                    f"Processing page {i + 1}/{total_images}"
                )
                
                # Extract text
                text = self.extract_text_from_image(image)
                
                if text:
                    # Extract questions
                    questions = self.identify_question_patterns(text)
                    
                    # Add page number
                    for q in questions:
                        q['page'] = i + 1
                    
                    all_questions.extend(questions)
                    self.log(f"Found {len(questions)} questions on page {i + 1}")
            
            # Post-process all questions
            self.update_progress(80, "Post-processing questions...")
            all_questions = self.post_process_questions(all_questions)
            
            # Sort by question number
            all_questions.sort(key=lambda x: (x['page'], x['number']))
            
            # Create DataFrame
            self.update_progress(90, "Creating Excel file...")
            df = self.create_dataframe(all_questions)
            
            # Save to Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            self.update_progress(100, f"Extraction complete! Found {len(all_questions)} questions")
            self.log(f"Excel file saved to: {output_path}")
            
            return True, f"Successfully extracted {len(all_questions)} questions"
            
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            return False, error_msg
    
    def create_dataframe(self, questions: List[Dict]) -> pd.DataFrame:
        """Create a pandas DataFrame from extracted questions"""
        data = []
        
        for q in questions:
            row = {
                'Question Number': q['number'],
                'Page': q['page'],
                'Question': q['question'],
                'Type': q['type'],
                'Has Formula': q['has_formula'],
                'Character Count': q['char_count'],
                'Word Count': q['word_count']
            }
            
            # Add options if present
            for opt in ['A', 'B', 'C', 'D']:
                row[f'Option {opt}'] = q['options'].get(opt, '')
            
            data.append(row)
        
        return pd.DataFrame(data)


class QuestionExtractorGUI:
    """GUI for the Question Extractor"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("JEE Question Paper Extractor")
        self.root.geometry("900x700")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.pdf_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.extractor = None
        self.extraction_thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="JEE Question Paper Extractor",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # PDF Selection
        ttk.Label(main_frame, text="PDF File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        pdf_entry = ttk.Entry(main_frame, textvariable=self.pdf_path, width=50)
        pdf_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(
            main_frame, 
            text="Browse", 
            command=self.select_pdf
        ).grid(row=1, column=2, pady=5)
        
        # Output Selection
        ttk.Label(main_frame, text="Output Excel:").grid(row=2, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(main_frame, textvariable=self.output_path, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        ttk.Button(
            main_frame, 
            text="Browse", 
            command=self.select_output
        ).grid(row=2, column=2, pady=5)
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="Extraction Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # DPI Setting
        ttk.Label(settings_frame, text="Scan DPI:").grid(row=0, column=0, sticky=tk.W)
        self.dpi_var = tk.IntVar(value=300)
        dpi_spinbox = ttk.Spinbox(
            settings_frame, 
            from_=150, 
            to=600, 
            textvariable=self.dpi_var,
            width=10
        )
        dpi_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(settings_frame, text="(Higher = Better quality but slower)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Language Setting
        ttk.Label(settings_frame, text="Language:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lang_var = tk.StringVar(value="en")
        lang_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.lang_var, 
            values=["en", "hi", "mixed"],
            width=10,
            state="readonly"
        )
        lang_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Progress Bar
        ttk.Label(main_frame, text="Progress:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var, 
            maximum=100
        )
        self.progress_bar.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Progress Label
        self.progress_label = ttk.Label(main_frame, text="Ready")
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        # Log Text Area
        log_frame = ttk.LabelFrame(main_frame, text="Extraction Log", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=15
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        self.extract_btn = ttk.Button(
            button_frame, 
            text="Start Extraction", 
            command=self.start_extraction,
            style="Accent.TButton"
        )
        self.extract_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame, 
            text="Stop", 
            command=self.stop_extraction,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Clear Log", 
            command=self.clear_log
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Exit", 
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)
    
    def select_pdf(self):
        """Select PDF file"""
        filename = filedialog.askopenfilename(
            title="Select JEE Question Paper PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)
            # Auto-generate output path
            base = os.path.splitext(filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path.set(f"{base}_extracted_{timestamp}.xlsx")
    
    def select_output(self):
        """Select output Excel file"""
        filename = filedialog.asksaveasfilename(
            title="Save Extracted Questions As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, value, message=""):
        """Update progress bar and label"""
        self.progress_var.set(value)
        if message:
            self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
    
    def start_extraction(self):
        """Start the extraction process"""
        # Validate inputs
        if not self.pdf_path.get():
            messagebox.showerror("Error", "Please select a PDF file")
            return
        
        if not os.path.exists(self.pdf_path.get()):
            messagebox.showerror("Error", "Selected PDF file does not exist")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file")
            return
        
        # Disable buttons
        self.extract_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Clear previous log
        self.clear_log()
        
        # Start extraction in separate thread
        self.extraction_thread = threading.Thread(target=self.run_extraction)
        self.extraction_thread.start()
    
    def run_extraction(self):
        """Run the extraction process"""
        try:
            # Create extractor
            self.extractor = QuestionExtractor(
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )
            
            # Run extraction
            success, message = self.extractor.extract_from_pdf(
                self.pdf_path.get(),
                self.output_path.get()
            )
            
            # Show result
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Success", message))
                self.root.after(0, lambda: self.open_output_folder())
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", message))
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.log_message(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            # Re-enable buttons
            self.root.after(0, self.extraction_complete)
    
    def extraction_complete(self):
        """Reset UI after extraction"""
        self.extract_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_progress(0, "Ready")
    
    def stop_extraction(self):
        """Stop the extraction process"""
        if self.extraction_thread and self.extraction_thread.is_alive():
            self.log_message("Stopping extraction...")
            # Note: Proper thread stopping would require more sophisticated handling
            # This is a simplified version
            self.extraction_complete()
    
    def open_output_folder(self):
        """Open the folder containing the output file"""
        output_dir = os.path.dirname(self.output_path.get())
        if os.path.exists(output_dir):
            import platform
            if platform.system() == 'Windows':
                os.startfile(output_dir)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{output_dir}"')
            else:  # Linux
                os.system(f'xdg-open "{output_dir}"')


def main():
    """Main entry point"""
    # Check for required packages
    required_packages = {
        'tkinter': 'tkinter',
        'pandas': 'pandas',
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'fitz': 'PyMuPDF',
        'pix2text': 'pix2text',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = QuestionExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # Add missing import
    import io
    main()
