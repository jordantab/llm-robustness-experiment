import pandas as pd 
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import argparse

'''
         bm  Accuracy  F1_Score    Recall  Precision
0       A++  0.531362  0.454286  0.573271   0.453480
1        PB  0.579809  0.657514  0.627408   0.456427
2       DDX  0.573197  0.397910  0.436914   0.462220
3        FK  0.419269  0.431112  0.345701   0.509963
4   A++_ahp  0.594199  0.662525  0.557539   0.451572
5    PB_ahp  0.583166  0.548958  0.514332   0.534676
6   DDX_ahp  0.531344  0.390982  0.336464   0.563217
7    FK_ahp  0.350607  0.649425  0.432308   0.411694
8   A++_icr  0.352457  0.696394  0.381133   0.381873
9    PB_icr  0.583958  0.422228  0.386226   0.361798
10  DDX_icr  0.302373  0.412781  0.362364   0.556926
11   FK_icr  0.384559  0.381491  0.320060   0.580906
'''

def get_metric_pairs(csv_file, metric):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Define pairs for base, ahp, and icr
    base_pairs = [["A++", "DDX"], ["A++", "FK"], ["PB", "DDX"], ["PB", "FK"]]
    ahp_pairs = [["A++_ahp", "DDX_ahp"], ["A++_ahp", "FK_ahp"], ["PB_ahp", "DDX_ahp"], ["PB_ahp", "FK_ahp"]]
    icr_pairs = [["A++_icr", "DDX_icr"], ["A++_icr", "FK_icr"], ["PB_icr", "DDX_icr"], ["PB_icr", "FK_icr"]]

    # Initialize lists for x and y
    x = []
    y = []

    # Helper function to extract (x, y) for pairs
    def extract_pairs(pairs):
        pairs_xy = []
        for pair in pairs:
            xy_pair = [
                df.loc[df['bm'] == pair[0], metric].values[0],
                df.loc[df['bm'] == pair[1], metric].values[0]
            ]
            pairs_xy.append(xy_pair)
        return pairs_xy

    # Get accuracy pairings for all types
    base_accuracy = extract_pairs(base_pairs)
    ahp_accuracy = extract_pairs(ahp_pairs)
    icr_accuracy = extract_pairs(icr_pairs)

    # Combine all pairs into x and y
    for pair in base_accuracy + ahp_accuracy + icr_accuracy:
        x.append(pair[0])
        y.append(pair[1])

    return x, y


def lin_fit(x,y):
    x_lin = np.array(x)
    y_lin = np.array(y)

    # Calculate the coefficients of the best fit line
    m, b = np.polyfit(x_lin, y_lin, 1)

    return m, b

def do_pearson(x,y):
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value

def calc_corr_and_plot_all(csv1, csv2, metric):
    # Get x and y for each CSV
    x1, y1 = get_metric_pairs(csv1, metric)
    x2, y2 = get_metric_pairs(csv2, metric)

    # Perform linear fits
    m1, b1 = lin_fit(x1, y1)
    m2, b2 = lin_fit(x2, y2)

    # Compute Pearson correlations
    corr1, pval1 = do_pearson(x1, y1)
    corr2, pval2 = do_pearson(x2, y2)

    # Create best fit lines
    y_fit1 = m1 * np.array(x1) + b1
    y_fit2 = m2 * np.array(x2) + b2

    # Plot the data and best fit lines
    plt.scatter(x1, y1, color='blue')
    plt.plot(x1, y_fit1, color='blue', linestyle='--', label=f'Mixtral:8x7B \n y = {m1:.2f}x + {b1:.2f}')
    
    plt.scatter(x2, y2, color='green')
    plt.plot(x2, y_fit2, color='green', linestyle='--', label=f"llama2-7B \n y = {m2:.2f}x + {b2:.2f}")


    # Add labels and legend
    plt.xlabel(f'Adversarial Robustness ({metric})')
    plt.ylabel(f'OOD Robustness ({metric})')
    plt.title(f'Relationship of Adversarial vs OOD Robustness')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    return {
        "Llama2:7B": {"correlation": corr1, "p_value": pval1},
        "Mixtral:8x7B": {"correlation": corr2, "p_value": pval2}
    }

def main():
    parser = argparse.ArgumentParser(description="Script to perform tasks with a specified model.")

    parser.add_argument(
        "--metric", 
        type=str, 
        required=True, 
        help="The metric to use: Accuracy, F1_Score, Recall or Precision"
    )

    args = parser.parse_args()
    metric = args.metric
    corr_dict = calc_corr_and_plot_all('llama.csv', 'mixtral.csv', metric)