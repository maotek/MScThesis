
import os
import pandas as pd

def generate_tables():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'evaluate_results')
    output_dir = results_dir
    
    data = []

    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_dir, filename)
            
            experiment_name = filename.replace("dsec_validation_", "").replace("mvsec_validation_", "").replace(".csv", "")
            
            df = pd.read_csv(filepath)
            
            # Find the row with _RMS_linear and get the MEAN value
            try:
                rmse_linear_row = df[df['METRIC'] == '_RMS_linear']
                if not rmse_linear_row.empty:
                    rmse_mean = rmse_linear_row['MEAN'].iloc[0]
                else:
                    rmse_mean = '-'
            except (KeyError, IndexError):
                rmse_mean = '-'

            dataset_type = 'dsec' if 'dsec' in filename else 'mvsec' if 'mvsec' in filename else 'unknown'

            found = False
            for item in data:
                if item['experiment'] == experiment_name:
                    if dataset_type == 'dsec':
                        item['dsec_rmse'] = rmse_mean
                    elif dataset_type == 'mvsec':
                        item['mvsec_rmse'] = rmse_mean
                    found = True
                    break
            
            if not found:
                new_item = {'experiment': experiment_name, 'dsec_rmse': '-', 'mvsec_rmse': '-'}
                if dataset_type == 'dsec':
                    new_item['dsec_rmse'] = rmse_mean
                elif dataset_type == 'mvsec':
                    new_item['mvsec_rmse'] = rmse_mean
                data.append(new_item)

    # Sort data by experiment name
    data = sorted(data, key=lambda x: x['experiment'])

    # --- Generate Markdown Table ---
    md_content = "| Experiment | DSEC RMSE | MVSEC RMSE |\n"
    md_content += "|------------|-------------------------|--------------------------|\n"
    for item in data:
        dsec_rmse = f"{item['dsec_rmse']:.4f}" if isinstance(item['dsec_rmse'], float) else item['dsec_rmse']
        mvsec_rmse = f"{item['mvsec_rmse']:.4f}" if isinstance(item['mvsec_rmse'], float) else item['mvsec_rmse']
        md_content += f"| {item['experiment']} | {dsec_rmse} | {mvsec_rmse} |\n"

    with open(os.path.join(results_dir, "summary_table.md"), "w") as f:
        f.write(md_content)

    # --- Generate LaTeX Table ---
    latex_content = "% Add \\usepackage{longtable} to your LaTeX preamble\n"
    latex_content += "\\begin{longtable}{|l|c|c|}\n"
    latex_content += "\\caption{Summary of RMSE linear results.}\n"
    latex_content += "\\label{tab:rmse_summary} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\endfirsthead\n"
    latex_content += "\\multicolumn{3}{c}%\n"
    latex_content += "{\\tablename\\ \\thetable{} -- continued from previous page} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "Experiment & DSEC RMSE & MVSEC RMSE \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\endhead\n"
    latex_content += "\\hline\n"
    latex_content += "\\multicolumn{3}{r}{{Continued on next page}} \\\\\n"
    latex_content += "\\endfoot\n"
    latex_content += "\\endlastfoot\n"
    latex_content += "Experiment & DSEC RMSE & MVSEC RMSE \\\\\n\\hline\n"
    for item in data:
        experiment_name_escaped = item['experiment'].replace('_', '\\_')
        dsec_rmse = f"{item['dsec_rmse']:.4f}" if isinstance(item['dsec_rmse'], float) else item['dsec_rmse']
        mvsec_rmse = f"{item['mvsec_rmse']:.4f}" if isinstance(item['mvsec_rmse'], float) else item['mvsec_rmse']
        latex_content += f"{experiment_name_escaped} & {dsec_rmse} & {mvsec_rmse} \\\\\n"
    
    latex_content += "\\hline\n\\end{longtable}"

    with open(os.path.join(results_dir, "summary_table.tex"), "w") as f:
        f.write(latex_content)

    print("Summary tables created in 'evaluate_results' directory.")

if __name__ == "__main__":
    generate_tables()
