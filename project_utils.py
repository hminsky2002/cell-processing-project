import pandas as pd

def save_results(results_list, output_path='results/component_comparison.csv'):
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
