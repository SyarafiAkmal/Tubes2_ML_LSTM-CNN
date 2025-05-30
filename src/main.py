from cnn_analysis import *
from rnn_analysis import *
from lstm_analysis import *


def main():
    # Analyze each model type
    analyze_cnn_hyperparameters()

    analyze_rnn_hyperparameters()

    analyze_lstm_hyperparameters()

if __name__ == "__main__":
    main()