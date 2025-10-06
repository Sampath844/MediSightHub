# 🏥 DoctorAI Medical Platform

An advanced AI-powered medical imaging analysis and collaboration platform built with Streamlit and Google's Gemini AI.

## ✨ Features

### 🔬 Medical Image Analysis
- **Multi-format Support**: DICOM, NIfTI, JPEG, PNG
- **AI-Powered Analysis**: Comprehensive medical image interpretation using Gemini 2.5 Pro
- **XAI (Explainable AI)**: Visual heatmaps showing AI focus areas
- **Attention Heatmaps**: Advanced visualization of model attention
- **Diagnosis Integration**: Upload and compare with radiologist reports

### 💬 Collaboration Hub
- **Real-time Chat**: Medical professionals can discuss cases
- **Image Upload**: Share medical images in chat for collaborative analysis
- **AI Assistant**: Gemini-powered medical AI participates in discussions
- **Multi-user Support**: Multiple doctors can join case discussions

### ❓ Report Q&A System
- **Patient-Friendly**: Ask questions about medical reports in plain language
- **Image OCR**: Upload handwritten prescriptions and reports
- **Gemini Vision**: Analyze medical documents and images
- **Context-Aware**: AI understands your previous reports and findings

### 📊 Analytics Dashboard
- **Statistics**: Track analysis patterns and common findings
- **Insights**: Visualize medical data trends
- **History**: Access previous analyses and reports

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Edit .env file and add your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 🔧 Configuration

Edit your `.env` file:

```env
# Required: Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Email for PubMed searches
ENTREZ_EMAIL=your_email@example.com
```

## 📋 Usage Guide

### Medical Image Analysis

1. **Upload Images**: Drag and drop medical images (DICOM, JPEG, PNG, NIfTI)
2. **Enable XAI**: Toggle explainable AI for heatmap generation
3. **Add Diagnosis**: Upload radiologist reports for comparison
4. **Generate Reports**: Download comprehensive PDF reports

### Collaboration

1. **Create Room**: Start a new case discussion
2. **Invite Team**: Share room ID with colleagues
3. **Upload Images**: Share medical images in chat
4. **AI Assistance**: Ask questions to the AI assistant

### Patient Q&A

1. **Select Report**: Choose from analyzed medical reports
2. **Create Session**: Start a new Q&A session
3. **Ask Questions**: Get explanations in simple language
4. **Upload Documents**: Share prescriptions or handwritten notes

## 🔒 Privacy & Security

- **Local Processing**: All data stored locally by default
- **No Data Transmission**: Images processed through secure API calls
- **Environment Variables**: Secure API key management
- **Session Isolation**: Separate user sessions

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` is set in `.env`
2. **Import Errors**: Install all requirements with `pip install -r requirements.txt`
3. **Image Processing**: Install optional dependencies for DICOM/NIfTI support
4. **Memory Issues**: Use smaller images or increase system memory

---

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. Always consult with qualified healthcare professionals for medical decisions. The AI analysis should not replace professional medical judgment.
