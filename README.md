# Face-Anti-Spoofing
Face Anti-Spoofing Using Deep-Pixel-wise-Binary-Supervision

- Implementation of paper: https://arxiv.org/pdf/1907.04047v1.pdf
- This project is based on: https://github.com/voqtuyen/deep-pix-bis-pad.pytorch
## Dependencies:
Run `pip install -r requirements.txt`
## Dataset:
We are using: http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html
Split images into 2 folders (fake and real):
- Modify data folder path in dataPreprocess.py
- Run `python dataPrepocess.py' 
- The result is two csv files: train_compose.csv and test_compose.csv (split with ratio 0.75)
## Test:
Run `python gui.py`
