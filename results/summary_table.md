| Model | Split | Val acc | Val macro-F1 | Test acc | Test macro-F1 (95% CI) | Train time |
|---|---|---|---|---|---|---|
| MLP (TF-IDF) | temporal | 0.8698 | 0.8727 | 0.8762 | 0.8771 (0.8687 to 0.8843) | 24 s |
| MLP (TF-IDF) | random | 0.8864 | 0.8874 | 0.8851 | 0.8860 (0.8786 to 0.8929) | 21 s |
| BiLSTM | temporal | 0.8648 | 0.8683 | 0.8687 | 0.8690 (0.8604 to 0.8764) | 34 min |
| BiLSTM | random | 0.8852 | 0.8860 | 0.8835 | 0.8840 (0.8772 to 0.8908) | 30 min |
| DistilBERT | temporal | 0.8901 | 0.8941 | 0.8913 | 0.8921 (0.8849 to 0.8988) | 32 min |
| DistilBERT | random | 0.9023 | 0.9029 | 0.9060 | 0.9066 (0.9000 to 0.9128) | 43 min |
