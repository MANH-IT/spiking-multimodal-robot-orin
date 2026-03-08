@echo off
echo ============================================================
echo  Cai dat dependencies - NCKH 2026 Multi-Modal Robot AI
echo ============================================================

echo.
echo [1/5] Core AI frameworks...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/5] Spiking Neural Networks (snntorch)...
pip install snntorch

echo.
echo [3/5] Vietnamese NLP (underthesea)...
pip install underthesea pyvi

echo.
echo [4/5] Multilingual embeddings + language detection...
pip install sentence-transformers langdetect

echo.
echo [5/5] Training utilities...
pip install tqdm tensorboard

echo.
echo ============================================================
echo  Kiem tra cai dat...
echo ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import snntorch; print(f'snntorch: {snntorch.__version__}')" 2>nul || echo "snntorch: can cai lai"
python -c "from underthesea import word_tokenize; print('underthesea: OK')" 2>nul || echo "underthesea: can cai lai"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers: OK')" 2>nul || echo "sentence-transformers: can cai lai"
python -c "from langdetect import detect; print('langdetect: OK')" 2>nul || echo "langdetect: can cai lai"

echo.
echo ============================================================
echo  Hoan thanh! Chay thu:
echo    python nlp_system/models/spiking_language_model.py
echo    python nlp_system/models/vietnamese_graph_parser.py
echo    python nlp_system/models/cross_lingual_adapter.py
echo    python nlp_system/train_nlp.py
echo ============================================================
pause
