# Advanced OCR Bill Title Extractor

This project extracts **merchant / bill titles from receipt or invoice images** using a hybrid OCR pipeline combining **PaddleOCR** and **Tesseract**.

The system is optimized for **speed, robustness, and noisy receipt images**, and automatically detects the most likely **business name at the top of the receipt**.

---

## Features

* Hybrid OCR system

  * **PaddleOCR** for high-accuracy title detection
  * **Tesseract OCR** for full body text extraction
* Automatic **receipt region detection**
* **Skew detection and correction**
* Advanced **image preprocessing**
* Intelligent **title scoring algorithm**
* Multi-threaded OCR execution
* Automatic **candidate ranking**
* Exports results to **CSV**

---

## Project Workflow

1. Load receipt image
2. Detect receipt region
3. Preprocess image (denoise + contrast enhancement)
4. Detect and correct skew
5. Split receipt into:

   * **Top section** → merchant title
   * **Body section** → receipt text
6. Run OCR engines:

   * PaddleOCR → title detection
   * Tesseract → body extraction
7. Score text lines to identify the **best title candidate**
8. Save results to CSV

---

## Repository Structure

```
Advanced-OCR-Recognition
│
├── README.md
├── requirements.txt
│
└── OCR
    ├── Data
    │   └── images
    │       ├── receipt1.jpg
    │       ├── receipt2.png
    │
    └── Src
        └── bill_extractor.py
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Advanced-OCR-Recognition.git
cd Advanced-OCR-Recognition
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Required libraries:

```
opencv-python
numpy
pytesseract
pillow
paddleocr
```

### 3. Install Tesseract OCR

Download and install:

https://github.com/tesseract-ocr/tesseract

After installation ensure the command works:

```bash
tesseract --version
```

---

## Usage

Run the extractor on a folder containing receipt images.

```bash
python bill_extractor.py <image_folder>
```

Example:

```bash
python bill_extractor.py OCR/Data/images
```

You can also specify a custom output file:

```bash
python bill_extractor.py OCR/Data/images output.csv
```

---

## Output

The script generates a CSV file with the following fields:

| Column     | Description                |
| ---------- | -------------------------- |
| file       | Image filename             |
| title      | Extracted merchant name    |
| candidates | Top candidate text lines   |
| body_text  | OCR text from receipt body |
| time_sec   | Processing time per image  |

Example output:

```
file,title,candidates,body_text,time_sec
receipt1.jpg,STARBUCKS,"STARBUCKS | STARBUCKS COFFEE",...,0.92
receipt2.jpg,MCDONALDS,"MCDONALDS | McDonalds Restaurant",...,1.04
```

---

## OCR Strategy

### Title Detection

The algorithm evaluates text lines based on:

* Position on receipt
* Text size
* Character case patterns
* Center alignment
* Word count
* Confidence score
* Venue keywords

### Noise Filtering

The system removes common receipt noise such as:

* totals
* payment lines
* addresses
* phone numbers
* timestamps
* transaction IDs

---

## Performance Optimizations

* Global PaddleOCR model loaded **once**
* Shared **ThreadPoolExecutor**
* Image resizing for faster OCR
* Skew detection on downscaled image
* Region cropping before OCR

Average processing speed:

```
~0.7 – 1.5 seconds per receipt
```

(depending on image size and CPU)

---

## Supported Image Formats

```
.jpg
.jpeg
.png
.webp
.tiff
.bmp
```

---

## Example Use Cases

* Receipt processing pipelines
* Expense automation
* Financial document digitization
* Merchant detection from bills
* OCR research projects

---

## License

MIT License

---

## Author

Advanced OCR pipeline built for **high-accuracy receipt title extraction using hybrid OCR systems**.
