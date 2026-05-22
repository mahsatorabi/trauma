# Mapping the Architecture of Childhood Trauma Research (2019–2026)

Temporal evolution, emerging gaps, and cross-method thematic validation of Web of Science literature on childhood trauma (2019–2026).

## Data

Place your WoS export as `data.csv` in the project root, then run preprocessing. Large corpus files are not stored in this repository (GitHub file-size limits).

## Requirements

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

## Pipeline

Run in order:

```bash
python preprocess.py
python rq1.py
python rq2.py
python rq3.py
python rq4.py --nltk
python rq5.py
python rq6.py --post
python draw_pipeline.py
```

- **RQ1:** Master BERTopic model, temporal trends, shift drivers  
- **RQ2:** Emerging themes, gaps, strategic diagram (requires RQ1)  
- **RQ3:** Prevalence and composition across four periods (P1–P4)  
- **RQ4:** Cross-paradigm robustness and noun-phrase triangulation  
- **RQ5:** Intellectual theme network (keyword coupling)  
- **RQ6:** LDA on peer-reviewed 2020–2025 subset  

Outputs are written to `rq1_output/` … `rq6_output/`.

## Citation

If you use this code or methods, please cite the associated manuscript (forthcoming).
