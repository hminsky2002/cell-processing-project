import pandas as pd

def save_cell_results(results_list, output_path='results/component_comparison.csv'):
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)
        return output_path



def analyze_cell_results(csv_path: str):
    df = pd.read_csv(csv_path)
    differences = df['difference']

    stats = {
        'mean': differences.mean(),
        'std': differences.std(),
        'min': differences.min(),
        'max': differences.max(),
        'median': differences.median()
    }

    stats_path = f'results/{csv_path.split("/")[-1].split(".")[0]}_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("Results Analysis:\n")
        f.write(f"Mean difference: {stats['mean']:.2f}\n")
        f.write(f"Std deviation: {stats['std']:.2f}\n")
        f.write(f"Min difference: {stats['min']}\n")
        f.write(f"Max difference: {stats['max']}\n")
        f.write(f"Median difference: {stats['median']:.2f}\n")