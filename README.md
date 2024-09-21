# Information_Fast_Checking
Verify the accuracy of a statement within a paragraph (vietnamese)

## 1. Installation
`conda` virtual environment is recommended
```
conda create -n fast_checking_env python=3.9 -y
conda activate fast_checking_env
```
After that you access [Start Locally Pytorch](https://pytorch.org/get-started/locally/) to install pytorch.
```
pip install -r requirements.txt
```
## 2. Usage
First:
```command
cd Information_Fast_Checking
conda activate fast_checking_env
```
### 1. Use local
```python
streamlit run app.py
```
### 2. Use ngrok to share