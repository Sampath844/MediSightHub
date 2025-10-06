# 🚀 DoctorAI Setup Instructions

## Complete Setup Guide

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Gemini API key

### 2. Installation Steps

```bash
# 1. Extract the ZIP file
unzip doctorai-platform.zip
cd doctorai-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
# Edit .env file and add your Gemini API key
```

### 3. Environment Configuration

Edit your `.env` file:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
ENTREZ_EMAIL=your_email@example.com
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Access the Platform

Open your browser and navigate to `http://localhost:8501`

## ✅ Enhanced Features Implemented

1. **✅ Diagnosis File Upload**: Users can now upload radiologist/lab center diagnosis files
2. **✅ Enhanced XAI**: Improved heatmap generation with attention mapping
3. **✅ Image Upload in Chat**: Gemini Vision support for image analysis in chat rooms and Q&A
4. **✅ Modern UI**: React-like design with animations and professional styling
5. **✅ Environment API Key**: API key now accessed from .env file, no manual input required

## 🔧 API Key Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`
4. Restart the application

## 📞 Support

For issues or questions, check the troubleshooting section in README.md
