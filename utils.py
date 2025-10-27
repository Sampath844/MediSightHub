
import io
import base64
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RPImage, Table
from reportlab.platypus import PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
import requests
from io import BytesIO
from Bio import Entrez
import json
import re
from huggingface_hub import InferenceClient
import nibabel as nib
import pydicom
import PyPDF2
import docx
import fitz  # PyMuPDF for better PDF processing

Entrez.email = "sampat252004@gmail.com"

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

def process_diagnosis_file(uploaded_file):
    """Process uploaded diagnosis files (PDF, TXT, DOC, images with OCR)"""
    if not uploaded_file:
        return ""

    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'txt':
            # Simple text file
            content = uploaded_file.read().decode('utf-8')
            return content

        elif file_extension == 'pdf':
            # PDF file processing with PyMuPDF (better than PyPDF2)
            content = ""
            try:
                # Try with PyMuPDF first (better)
                pdf_bytes = uploaded_file.read()
                pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    content += page.get_text()
                pdf_document.close()
            except:
                # Fallback to PyPDF2
                uploaded_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    content += page.extract_text()
            return content

        elif file_extension in ['doc', 'docx']:
            # Word document processing
            uploaded_file.seek(0)
            doc = docx.Document(uploaded_file)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content

        elif file_extension in ['jpg', 'jpeg', 'png']:
            # Image OCR processing using Gemini Vision
            return process_image_ocr(uploaded_file)

        else:
            return f"Unsupported file format: {file_extension}"

    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_image_ocr(uploaded_file):
    """Process image files for OCR using Gemini Vision"""
    try:
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return "Error: Gemini API key not found in environment"

        image = Image.open(uploaded_file)

        # Use Gemini Vision for OCR
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = [
            "Extract all text from this medical document/report. Please provide the text exactly as written, maintaining the structure and format. If this is a medical report or diagnosis, extract all findings, impressions, and recommendations. Ignore any personal information",
            Image.open(io.BytesIO(img_bytes))
        ]

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error processing image with OCR: {str(e)}"

def process_file(uploaded_file):
    """Process uploaded medical image files with enhanced capabilities"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file).convert("RGB")
        return {"type": "image", "data": image, "array": np.array(image)}

    elif file_extension in ['dcm'] and PYDICOM_AVAILABLE:
        try:
            bytes_data = uploaded_file.getvalue()
            with io.BytesIO(bytes_data) as dcm_bytes:
                dicom = pydicom.dcmread(dcm_bytes)
                image_array = dicom.pixel_array

                # Enhanced DICOM processing
                if len(image_array.shape) == 3:
                    # Multi-slice, take middle slice
                    image_array = image_array[image_array.shape[0] // 2]

                # Proper windowing for medical images
                if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                    window_center = float(dicom.WindowCenter[0] if isinstance(dicom.WindowCenter, list) else dicom.WindowCenter)
                    window_width = float(dicom.WindowWidth[0] if isinstance(dicom.WindowWidth, list) else dicom.WindowWidth)

                    lower = window_center - window_width / 2
                    upper = window_center + window_width / 2
                    image_array = np.clip(image_array, lower, upper)

                # Convert to 8-bit for display
                image_array = ((image_array - image_array.min())/(image_array.max() - image_array.min()) * 255).astype(np.uint8)

                return {
                    "type": "dicom",
                    "data": Image.fromarray(image_array),
                    "array": image_array,
                    "metadata": {
                        "patient_id": getattr(dicom, 'PatientID', 'Unknown'),
                        "study_date": getattr(dicom, 'StudyDate', 'Unknown'),
                        "modality": getattr(dicom, 'Modality', 'Unknown')
                    }
                }
        except Exception as e:
            print(f"Error processing DICOM: {e}")
            return None

    elif file_extension in ["nii", "nii.gz"] and NIBABEL_AVAILABLE:
        try:
            bytes_data = uploaded_file.getvalue()
            with io.BytesIO(bytes_data) as nii_bytes:
                temp_path = f"temp_{uuid.uuid4()}.nii.gz"
                with open(temp_path, "wb") as f:
                    f.write(bytes_data)

                img = nib.load(temp_path)
                image_array = img.get_fdata()

                # Enhanced NIfTI processing
                if image_array.ndim == 3:
                    mid_slice = image_array.shape[2] // 2
                    slice_img = image_array[:, :, mid_slice]
                elif image_array.ndim == 4:
                    # 4D data, take middle slice of middle volume
                    mid_vol = image_array.shape[3] // 2
                    mid_slice = image_array.shape[2] // 2
                    slice_img = image_array[:, :, mid_slice, mid_vol]
                else:
                    slice_img = image_array

                # Normalize for display
                slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
                pil_img = Image.fromarray(slice_img)

                os.remove(temp_path)

                return {
                    "type": "nifti",
                    "data": pil_img,
                    "array": image_array,
                    "metadata": {
                        "shape": image_array.shape,
                        "affine": img.affine.tolist(),
                        "header": str(img.header)
                    }
                }
        except Exception as e:
            print(f"Error processing NIfTI: {e}")
            return None

    elif file_extension in ['dcm', 'nii', 'nii.gz']:
        return {
            "type": "image",
            "data": Image.new('RGB', (400, 400), color='gray'),
            "array": np.zeros((400, 400, 3), dtype=np.uint8),
            "metadata": {
                "Warning": "Required libraries not installed for this file type",
                "Missing": "Install pydicom or nibabel to process this file"
            }
        }

    else:
        print("Unsupported file type or required library not available.")
        return None

def generate_heatmap(image_array):
    """Generate a heatmap overlay for XAI visualization"""
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array

    # Enhanced heatmap generation
    # Apply Gaussian blur for smoother heatmap
    blurred = cv2.GaussianBlur(gray_image, (15, 15), 0)

    # Apply gradient magnitude for edge detection
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize and convert to heatmap
    magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)

    # Create overlay
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(heatmap, 0.6, image_array, 0.4, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap)

import json
import re
import numpy as np
import cv2
from PIL import Image
import io
import os

def generate_visual_prompt_heatmap(image, findings, api_key):
    """
    Generates an AI attention heatmap using a structured visual prompting technique.

    This method asks the model to identify grid coordinates for each finding,
    parses the structured JSON output, and creates a more precise heatmap.

    Args:
        image (PIL.Image): The medical image to analyze.
        findings (list): A list of key medical findings.
        api_key (str): The Google Gemini API key.

    Returns:
        PIL.Image: An image with the attention heatmap overlaid, or the original image on failure.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        # --- 1. Enhanced Prompt Engineering ---
        # We define a 5x5 grid (A-E, 1-5) and ask the model to return JSON.
        # This is more robust than parsing natural language.
        prompt = f"""
        Analyze the provided medical image in the context of the following findings: {findings}.

        Imagine a 10x10 grid overlaid on the image (rows A-J, columns 1-10).
        For each of the key findings, identify the grid cells that contain the most prominent evidence of that finding.

        Your response MUST be a JSON object...
        ...
        Example JSON response:
        {{
          "locations": [
            {{
              "finding": "femur fracture",
              "cells": ["F5", "F6", "G5", "G6"]
            }}
          ]
        }}
        """

        # --- 2. Call the AI model ---
        response = model.generate_content([prompt, image])
        
        # --- 3. Parse the Structured JSON Output ---
        # Extract the JSON block from the model's response text.
        json_match = re.search(r'```json\n({.*?})\n```', response.text, re.DOTALL)
        if not json_match:
            # Fallback for when the model doesn't use markdown
            json_match = re.search(r'({.*?})', response.text, re.DOTALL)
        
        if not json_match:
            print("Error: Could not parse JSON from model response.")
            # Fallback to the original image if parsing fails
            return generate_heatmap(np.array(image))[0]

        locations_data = json.loads(json_match.group(1))
        
        # --- 4. Create Heatmap from Coordinates ---
        image_array = np.array(image.convert('RGB'))
        height, width, _ = image_array.shape
        attention_mask = np.zeros((height, width), dtype=np.float32)

        grid_size = 10
        cell_height, cell_width = height // grid_size, width // grid_size

        for loc in locations_data.get("locations", []):
            for cell in loc.get("cells", []):
                # Convert cell notation (e.g., "C3") to pixel coordinates
                row_char = cell[0].upper()
                col_num = int(cell[1:]) - 1
                
                if 'A' <= row_char <= 'J' and 0 <= col_num < grid_size:
                    row_idx = ord(row_char) - ord('A')
                    
                    y_start = row_idx * cell_height
                    y_end = y_start + cell_height
                    x_start = col_num * cell_width
                    x_end = x_start + cell_width
                    
                    # Add intensity to the corresponding mask region
                    attention_mask[y_start:y_end, x_start:x_end] = 1.0

        # --- 5. Smooth and Blend the Heatmap ---
        if np.max(attention_mask) > 0:
            # Apply a strong Gaussian blur to create a smooth, blob-like heatmap
            # The kernel size is proportional to the image size.
            blur_kernel_size = int(min(height, width) / 25)
            if blur_kernel_size % 2 == 0: blur_kernel_size += 1 # Kernel must be odd
            
            attention_mask = cv2.GaussianBlur(attention_mask, (blur_kernel_size, blur_kernel_size), 0)
            
            # Normalize the mask to range 0-1
            attention_mask = attention_mask / np.max(attention_mask)

            # Colorize the mask
            heatmap = (attention_mask * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Blend the heatmap with the original image
            overlay = cv2.addWeighted(image_array, 0.6, colored_heatmap, 0.4, 0)
            
            return Image.fromarray(overlay)
        else:
            # If no locations were found, return the original image
            print("Warning: Model returned no valid locations for the heatmap.")
            return image

    except Exception as e:
        print(f"Error generating visual prompt heatmap: {e}")
        # Fallback to a basic heatmap or original image in case of any error
        return generate_heatmap(np.array(image))[0]

def analyze_image(image, api_key, enable_xai=True, diagnosis_context=""):
    """
    Enhanced medical image analysis using Google's Gemini 2.5 Pro model with diagnosis context
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        # Enhanced prompt with diagnosis context
        base_prompt = [
            """You are an expert radiologist analyzing this medical image. Provide a comprehensive analysis including:

            1. **Image Type & Anatomy**: Identify the imaging modality and anatomical structures visible
            2. **Detailed Findings**: Describe all visible abnormalities, normal variants, and pathological changes
            3. **Clinical Significance**: Explain the medical importance of your findings
            4. **Differential Diagnosis**: List possible diagnoses based on imaging findings
            5. **Recommendations**: Suggest follow-up imaging or clinical correlation if needed

            Format your response with clear headings and be thorough in your analysis.
            Start with the body part and scan type as your first words.""",
            Image.open(io.BytesIO(img_bytes))
        ]

        # Add diagnosis context if provided
        if diagnosis_context:
            diagnosis_prompt = f"""

            **Additional Context**: A radiologist/lab center has provided this diagnosis:
            \"{diagnosis_context}\"

            Please compare your AI analysis with this professional diagnosis and note any:
            - Agreements between your findings and the provided diagnosis
            - Additional observations you can make
            - Any discrepancies (if any) and possible explanations

            This comparison will help validate and enhance the diagnostic accuracy."""

            base_prompt[0] = base_prompt[0] + diagnosis_prompt

        # Generate the analysis
        response = model.generate_content(base_prompt)
        analysis = response.text

        # Extract findings and keywords
        findings, keywords = extract_findings_and_keywords(analysis)

        return {
            "id": str(uuid.uuid4()),
            "analysis": analysis,
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
            "has_diagnosis_context": bool(diagnosis_context)
        }

    except ImportError:
        return {
            "id": str(uuid.uuid4()),
            "analysis": "Error: google-generativeai SDK not installed. Install with `pip install google-generativeai`.",
            "findings": [],
            "keywords": [],
            "date": datetime.now().isoformat(),
            "has_diagnosis_context": False
        }

    except Exception as e:
        return {
            "id": str(uuid.uuid4()),
            "analysis": f"Error analyzing image with Gemini: {e}",
            "findings": [],
            "keywords": [],
            "date": datetime.now().isoformat(),
            "has_diagnosis_context": False
        }
import json

def generate_differential_diagnosis(analysis_text, findings, api_key):
    """
    Generates a structured differential diagnosis with probabilities and evidence.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        # This is the key novel prompt
        findings_bullets = "\n- ".join(findings)
        prompt = f"""
        You are a master diagnostician AI. Based on the following comprehensive analysis and key findings, generate a structured differential diagnosis.

        --- Full Analysis Text ---
        {analysis_text}
        --- Key Findings ---
        - {findings_bullets}
        ---

        Your task is to return a JSON object containing a single key "diagnoses".
        This key should hold a list of potential diagnoses. For each diagnosis in the list, provide:
        1. "condition": The name of the medical condition.
        2. "probability": A float between 0.0 and 1.0 representing your confidence in this diagnosis. The sum of all probabilities should ideally be close to 1.0.
        3. "evidence": A JSON array of strings, where each string is a direct quote or a summarized finding from the provided analysis text that supports this specific diagnosis.
        4. "rationale": A brief sentence explaining why the evidence points to this condition.

        Example of the required JSON output format:
        {{
            "diagnoses": [
                {{
                    "condition": "Bacterial Pneumonia",
                    "probability": 0.75,
                    "evidence": ["Consolidation in the right lower lobe", "Presence of air bronchograms"],
                    "rationale": "The presence of consolidation is a classic sign of pneumonia."
                }},
                {{
                    "condition": "Pulmonary Edema",
                    "probability": 0.15,
                    "evidence": ["Mild pleural effusions noted", "Slight cardiomegaly"],
                    "rationale": "Pleural effusions can be associated with edema, but the localized consolidation is less typical."
                }},
                {{
                    "condition": "Malignancy",
                    "probability": 0.10,
                    "evidence": ["Irregular borders of the opacity"],
                    "rationale": "While less likely, the irregular border warrants consideration of a malignant process."
                }}
            ]
        }}
        """
        response = model.generate_content(prompt)
        # Clean and parse the JSON output from the model's text response
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)

    except Exception as e:
        print(f"Error generating differential diagnosis: {e}")
        return None

def extract_findings_and_keywords(analysis_text):
    """Extract findings and keywords from the analysis text with enhanced processing"""
    findings = []
    keywords = []

    # Enhanced pattern matching for medical findings
    finding_patterns = [
        r"finding[s]?:?\s*([^.\n]+)",
        r"impression:?\s*([^.\n]+)",
        r"conclusion[s]?:?\s*([^.\n]+)",
        r"diagnosis:?\s*([^.\n]+)",
        r"abnormalit(?:y|ies):?\s*([^.\n]+)",
        r"suggestion[s]?:?\s*([^.\n]+)"
    ]

    for pattern in finding_patterns:
        matches = re.finditer(pattern, analysis_text, re.IGNORECASE)
        for match in matches:
            finding = match.group(1).strip()
            if finding and len(finding) > 10:  # Filter short findings
                findings.append(finding)

    # Extract medical terminology and keywords
    medical_terms = [
        "opacity", "consolidation", "nodule", "mass", "lesion", "calcification", 
        "effusion", "atelectasis", "infiltrate", "fracture", "edema", "pneumothorax", 
        "emphysema", "fibrosis", "pleural", "cardiomegaly", "hyperinflation", 
        "collapse", "infection", "tumor", "malignancy", "benign", "cyst", "abscess", 
        "thickening", "stenosis", "dilatation", "hemorrhage", "ischemia", "infarct",
        "enhancement", "contrast", "signal", "intensity", "density", "artifact"
    ]

    # Clean and analyze text
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', analysis_text.lower())
    words = cleaned_text.split()

    # Extract medical keywords
    for word in words:
        if len(word) > 4 and word in medical_terms:
            keywords.append(word)

    # Extract potential new keywords
    for word in words:
        if len(word) > 6 and word not in keywords:
            # Check if it's likely a medical term (contains certain patterns)
            if any(suffix in word for suffix in ['osis', 'itis', 'oma', 'pathy', 'trophy']):
                keywords.append(word)

    # Remove duplicates and return
    findings = list(set(findings))
    keywords = list(set(keywords))

    return findings, keywords

# Storage and utility functions
def get_analysis_store():
    """Get the analysis store from storage with error handling"""
    try:
        if os.path.exists("analysis_store.json"):
            with open("analysis_store.json", "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        return {"analyses": []}
    except Exception as e:
        print(f"Error loading analysis store: {e}")
        return {"analyses": []}

def save_analysis(analysis_data, filename="unknown.jpg"):
    """Save analysis data to storage with enhanced error handling"""
    try:
        store = get_analysis_store()
        analysis_data["filename"] = filename
        store["analyses"].append(analysis_data)

        with open("analysis_store.json", "w") as f:
            json.dump(store, f, indent=2, default=str)

        return analysis_data
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return analysis_data

def get_analysis_by_id(analysis_id):
    """Get a specific analysis by ID"""
    store = get_analysis_store()
    for analysis in store["analyses"]:
        if analysis["id"] == analysis_id:
            return analysis
    return None

def get_latest_analyses(limit=10):
    """Get the most recent analyses"""
    store = get_analysis_store()
    sorted_analyses = sorted(store["analyses"], key=lambda x: x.get("date", ""), reverse=True)
    return sorted_analyses[:limit]

def generate_statistics_report():
    """Generate enhanced statistical report of findings"""
    store = get_analysis_store()

    if not store["analyses"]:
        return None

    # Enhanced statistics
    type_counts = {}
    keyword_counts = {}
    finding_counts = {}
    monthly_counts = {}

    for analysis in store["analyses"]:
        # Count by type
        analysis_type = analysis.get("type", "unknown")
        type_counts[analysis_type] = type_counts.get(analysis_type, 0) + 1

        # Count keywords
        for keyword in analysis.get("keywords", []):
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Count findings
        for finding in analysis.get("findings", []):
            finding_counts[finding] = finding_counts.get(finding, 0) + 1

        # Count by month
        try:
            date_str = analysis.get("date", "")
            if date_str:
                month = date_str[:7]  # YYYY-MM
                monthly_counts[month] = monthly_counts.get(month, 0) + 1
        except:
            pass

    return {
        "total_analyses": len(store["analyses"]),
        "analyses_by_type": type_counts,
        "top_keywords": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_findings": sorted(finding_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "monthly_distribution": monthly_counts,
        "avg_keywords_per_analysis": sum(len(a.get("keywords", [])) for a in store["analyses"]) / len(store["analyses"]) if store["analyses"] else 0,
        "avg_findings_per_analysis": sum(len(a.get("findings", [])) for a in store["analyses"]) / len(store["analyses"]) if store["analyses"] else 0
    }

def search_pubmed(keywords, max_results=5):
    """Search PubMed for relevant articles based on keywords"""
    if not keywords or not BIOPYTHON_AVAILABLE:
        return []

    query = ' AND '.join(keywords[:3])

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        results = Entrez.read(handle)

        if not results["IdList"]:
            return []

        fetch_handle = Entrez.efetch(db="pubmed", id=results["IdList"], rettype="medline", retmode="text")
        records = fetch_handle.read().split('\n\n')

        publications = []
        for record in records:
            if not record.strip():
                continue

            pub_data = {"id": "", "title": "", "journal": "", "year": ""}

            for line in record.split('\n'):
                if line.startswith('PMID- '):
                    pub_data["id"] = line[6:].strip()
                elif line.startswith('TI  - '):
                    pub_data["title"] = line[6:].strip()
                elif line.startswith('TA  - '):
                    pub_data["journal"] = line[6:].strip()
                elif line.startswith('DP  - '):
                    year_match = line[6:].strip().split()[0]
                    pub_data["year"] = year_match if year_match.isdigit() else "2024"

            if pub_data["id"]:
                publications.append(pub_data)

        return publications

    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []

def search_pubmed_enhanced(keywords, max_results_each=5):
    """Search PubMed for both relevant and recent articles"""
    if not keywords or not BIOPYTHON_AVAILABLE:
        return {"relevant": [], "recent": []}
    
    query = ' AND '.join(keywords[:3])
    
    try:
        # Most relevant articles (sorted by relevance)
        handle_rel = Entrez.esearch(db="pubmed", term=query, retmax=max_results_each, sort="relevance")
        results_rel = Entrez.read(handle_rel)
        
        # Most recent articles (sorted by publication date)
        handle_rec = Entrez.esearch(db="pubmed", term=query, retmax=max_results_each, sort="pub_date")
        results_rec = Entrez.read(handle_rec)
        
        def fetch_articles(id_list):
            if not id_list:
                return []
            
            fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
            records = fetch_handle.read().split('\n\n')
            
            publications = []
            for record in records:
                if not record.strip():
                    continue
                
                pub_data = {"id": "", "title": "", "journal": "", "year": ""}
                
                for line in record.split('\n'):
                    if line.startswith('PMID- '):
                        pub_data["id"] = line[6:].strip()
                    elif line.startswith('TI  - '):
                        pub_data["title"] = line[6:].strip()
                    elif line.startswith('TA  - '):
                        pub_data["journal"] = line[6:].strip()
                    elif line.startswith('DP  - '):
                        year_match = line[6:].strip().split()[0]
                        pub_data["year"] = year_match if year_match.isdigit() else "2024"
                
                if pub_data["id"]:
                    publications.append(pub_data)
            
            return publications
        
        return {
            "relevant": fetch_articles(results_rel["IdList"]),
            "recent": fetch_articles(results_rec["IdList"])
        }
    
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return {"relevant": [], "recent": []}

import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RPImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import io

def clean_text(text: str) -> str:
    """Remove markdown symbols (###, **, *) for neat PDF output"""
    text = re.sub(r"\*{1,3}", "", text)   # remove *, **, ***
    text = re.sub(r"#{1,6}", "", text)    # remove ###
    return text.strip()

def format_analysis_text(text):
    """Format AI analysis text into structured paragraphs & bullets"""
    text = clean_text(text)

    lines = []
    for part in re.split(r'(\d+\.\s)', text):  # split by 1., 2., 3.
        if part.strip():
            lines.append(part.strip())

    formatted = []
    current_section = ""

    for line in lines:
        if re.match(r'^\d+\.\s*', line):  # section number
            if current_section:
                formatted.append(current_section.strip())
            current_section = f"<b>{line}</b> "
        else:
            # turn ":" lists into bullets
            if ":" in line and any(word in line.lower() for word in ["bones", "joints", "soft", "vascular", "findings", "extremity"]):
                parts = line.split(":", 1)
                current_section += f"<br/><b>{parts[0]}:</b> {parts[1].strip()}"
            elif line.strip().startswith(("-", "•")):
                current_section += f"<br/>• {line.lstrip('-• ').strip()}"
            else:
                current_section += f"{line.strip()} "
    
    if current_section:
        formatted.append(current_section.strip())
    
    return formatted

def generate_report(data, include_references=True, add_images=True):
    """Generate structured PDF with cover, AI analysis, findings, images, heatmap, and PubMed refs"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # --- Styles ---
    title_style = ParagraphStyle(
        'Title', parent=styles["Heading1"], fontSize=28,
        textColor=colors.HexColor("#2E86C1"), alignment=1, spaceAfter=20
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#1F618D"), spaceBefore=12, spaceAfter=6
    )
    normal_style = ParagraphStyle("Normal", parent=styles["Normal"], fontSize=11, leading=15)
    highlight_style = ParagraphStyle("Highlight", parent=styles["Normal"], backColor=colors.whitesmoke,
                                     borderWidth=1, borderColor=colors.HexColor("#3498DB"),
                                     borderPadding=6, spaceAfter=8)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                                   textColor=colors.grey, alignment=1, spaceAfter=6)

    content = []

    # --- Cover Page ---
    content.append(Spacer(1, 150))
    content.append(Paragraph("📑 Medical Imaging Analysis Report", title_style))
    content.append(Spacer(1, 50))
    content.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Report ID:</b> {data['id']}", normal_style))
    if 'filename' in data:
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"<b>Image:</b> {data['filename']}", normal_style))
    content.append(Spacer(1, 250))
    content.append(Paragraph("Generated by AI Imaging Assistant", caption_style))
    content.append(PageBreak())

    # --- Metadata ---
    meta_table = Table([
        ["Date:", datetime.now().strftime('%Y-%m-%d %H:%M')],
        ["Report ID:", data["id"]],
        ["Image:", data.get("filename", "N/A")]
    ], colWidths=[80, 400])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D6EAF8")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(meta_table)
    content.append(Spacer(1, 14))

    # --- Images ---
    if add_images and data.get("image") and data.get("heatmap"):
        try:
            img_buf = io.BytesIO()
            data["image"].save(img_buf, format="PNG")
            img_buf.seek(0)
            rp_img = RPImage(img_buf, width=250, height=250)

            heatmap_buf = io.BytesIO()
            data["heatmap"].save(heatmap_buf, format="PNG")
            heatmap_buf.seek(0)
            rp_heatmap = RPImage(heatmap_buf, width=250, height=250)

            img_table = Table([[rp_img, rp_heatmap]], colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))

            captions = Table([[
                Paragraph("Original Scan", caption_style),
                Paragraph("Heatmap Overlay", caption_style)
            ]], colWidths=[270, 270])

            content.append(img_table)
            content.append(captions)
            content.append(Spacer(1, 16))
        except Exception as e:
            content.append(Paragraph(f"⚠️ Error adding images: {e}", normal_style))

    # --- Analysis (Structured) ---
    content.append(Paragraph("🔍 Analysis Results", subtitle_style))
    structured_analysis = format_analysis_text(data["analysis"]) or []
    for section in structured_analysis:
        content.append(Paragraph(section, normal_style))
        content.append(Spacer(1, 6))
    content.append(Spacer(1, 12))

    # --- Findings ---
    if data.get("findings"):
        content.append(Paragraph("🩺 Key Findings", subtitle_style))
        for idx, finding in enumerate(data["findings"], 1):
            content.append(Paragraph(f"{idx}. {clean_text(finding)}", highlight_style))
        content.append(Spacer(1, 12))

    # --- Keywords ---
    if data.get("keywords"):
        content.append(Paragraph("🔑 Keywords", subtitle_style))
        content.append(Paragraph(", ".join([clean_text(k) for k in data["keywords"]]), normal_style))
        content.append(Spacer(1, 12))

    # --- PubMed ---
       # --- PubMed References (Relevant and Recent) ---
    if include_references and data.get("keywords"):
        pubs = search_pubmed_enhanced(data["keywords"], max_results_each=5)
        
        # Most Relevant
        content.append(Paragraph("📚 Most Relevant Articles", subtitle_style))
        if pubs["relevant"]:
            for ref in pubs["relevant"]:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{ref['id']}/"
                line = (
                    f"<b>{ref['title']}</b><br/>"
                    f"<i>Journal:</i> {ref['journal']}, {ref['year']} "
                    f'(<a href="{url}">PMID:{ref["id"]}</a>)'
                )
                content.append(Paragraph(line, normal_style))
                content.append(Spacer(1, 6))
        else:
            content.append(Paragraph("No relevant articles found.", normal_style))
        content.append(Spacer(1, 12))
        
        # Most Recent
        content.append(Paragraph("🆕 Most Recent Articles", subtitle_style))
        if pubs["recent"]:
            for ref in pubs["recent"]:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{ref['id']}/"
                line = (
                    f"<b>{ref['title']}</b><br/>"
                    f"<i>Journal:</i> {ref['journal']}, {ref['year']} "
                    f'(<a href="{url}">PMID:{ref["id"]}</a>)'
                )
                content.append(Paragraph(line, normal_style))
                content.append(Spacer(1, 6))
        else:
            content.append(Paragraph("No recent articles found.", normal_style))
        content.append(Spacer(1, 12))

    doc.build(content)
    buffer.seek(0)
    return buffer
