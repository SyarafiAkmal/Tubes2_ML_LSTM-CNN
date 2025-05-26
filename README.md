# Tugas Besar 2 IF3270 - CNN dan RNN Implementation

Implementasi Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN/LSTM) from scratch untuk mata kuliah IF3270 Pembelajaran Mesin.

## Deskripsi Proyek

Proyek ini mengimplementasikan forward propagation untuk:
- **CNN**: Untuk klasifikasi gambar menggunakan dataset CIFAR-10
- **Simple RNN**: Untuk klasifikasi teks menggunakan dataset NusaX-Sentiment
- **LSTM**: Untuk klasifikasi teks menggunakan dataset NusaX-Sentiment


## Instalasi

1. Clone repository ini:
2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Cara Menjalankan

### Menjalankan Eksperimen Modular

\`\`\`bash
cd src

# Menjalankan semua eksperimen
python main.py --model all

# Menjalankan hanya LSTM
python main.py --model lstm

