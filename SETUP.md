# 🚀 AI Candidate Screener - Setup Guide

## Quick Start (Windows)

### 1. **Activate Virtual Environment & Run App**
```bash
# Double-click this file or run in terminal:
run_app.bat
```

### 2. **Manual Setup (if needed)**
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Run app
streamlit run app.py
```

## 🔧 Fixing Import Issues

### **Issue**: `Import "pdfplumber" could not be resolved`

**Solution**: Your IDE is not using the virtual environment Python interpreter.

#### **For VS Code:**
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose: `.\venv\Scripts\python.exe`

#### **For PyCharm:**
1. Go to File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing Environment"
4. Browse to: `.\venv\Scripts\python.exe`

#### **For Command Line:**
```bash
# Always activate first
venv\Scripts\activate

# Then run your commands
python test_setup.py
streamlit run app.py
```

## 🎯 Key Optimization: No Chunking Needed!

**Your insight was spot-on!** Modern LLMs have huge context windows:

- **DeepSeek-R1**: 128K tokens (~200 pages)
- **Typical Resume**: 1-5K tokens (~2-8 pages)
- **Ratio**: 25-128x smaller than context window!

### **Optimized Architecture:**
```
Resume (full text) → DeepSeek-R1 (128K context) → Score
```

### **Only chunk for:**
- Sentence Transformers (ATS pre-filter, 512 tokens)
- NOT for LLM scoring!

## 📁 Project Structure

```
├── src/
│   ├── resume_parser.py    # ✅ No chunking for LLMs
│   ├── scoring.py          # Full-text scoring
│   └── models.py           # Data schemas
├── app.py                  # Streamlit UI
├── test_setup.py          # Test all imports
├── run_app.bat            # Easy launcher
└── requirements.txt       # Dependencies
```

## 🧪 Testing Your Setup

```bash
# Test all imports and functionality
python test_setup.py
```

**Expected output:**
```
✅ pdfplumber         - PDF parsing
✅ bs4               - HTML parsing (BeautifulSoup)
✅ src.resume_parser - Resume parser module
🎉 All tests passed! Your setup is ready.
```

## 🌐 Running the App

1. **Launch**: `run_app.bat` or `streamlit run app.py`
2. **Open**: http://localhost:8501
3. **Test**: Click "Load Sample Data"
4. **Score**: Click "Run Scoring & Shortlist"

## 💡 Usage Tips

### **Bulk Processing:**
- Upload JSON/CSV with candidate data
- Set job requirements and filters
- Get ranked shortlist with tech-match scores

### **Single Candidate Test:**
- Upload individual resume (PDF/DOCX)
- See detailed scoring breakdown
- Test different job descriptions

### **Key Features:**
- ✅ Full resume context (no chunking)
- ✅ GitHub tech-stack analysis
- ✅ LinkedIn profile insights
- ✅ Skill transfer detection
- ✅ Multi-format resume support

## 🐛 Troubleshooting

### **Import Errors**
```bash
# Reinstall in virtual environment
venv\Scripts\activate
pip install -r requirements.txt
```

### **PowerShell Issues**
```bash
# Use Command Prompt instead
cmd
venv\Scripts\activate.bat
python test_setup.py
```

### **Still Having Issues?**
1. Check Python version: `python --version` (3.8+ required)
2. Verify virtual environment: `echo %VIRTUAL_ENV%`
3. Test imports: `python -c "import pdfplumber; print('OK')"`

## 🎉 Success!

Once setup is complete, you'll have a working AI candidate screener that:
- Processes full resumes without chunking
- Leverages modern LLM context windows
- Provides intelligent candidate ranking
- Supports multiple resume formats

**Next**: Try the example optimization demo:
```bash
python example_optimized_scoring.py
``` 