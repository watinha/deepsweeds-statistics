import pandas as pd, os, sys
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from mlxtend.evaluate import cochrans_q, mcnemar_table, mcnemar


RESULTS_FOLDER = './inferencia'
DATASETS = [ 'deepweeds', 'weed6c' ]
MODELS = [ 'efficientnetv2b0', 'efficientnetv2b1', 'efficientnetv2b2', 'efficientnetv2b3',
           'inceptionv3', 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large', 'nasnetmobile', 
           'resnet101v2', 'resnet50' ]


raw_results = {}
for model in MODELS:
    raw_results[model] = pd.DataFrame()

    for dataset in DATASETS:
        for i in range(1, 4):
            path = f'{RESULTS_FOLDER}/{dataset}/benchmark/predict_{dataset}_{model}_fold_{i}_val.csv'
            df = pd.read_csv(path)
            raw_results[model] = pd.concat([raw_results[model], df], ignore_index=True)
            print(raw_results[model].shape)

    print(raw_results[model].head())


OUTPUT_FOLDER = './resultados'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

with pd.ExcelWriter(f'{OUTPUT_FOLDER}/aggregated_metrics.xlsx') as writer:
    synthesized_results = pd.DataFrame(
            columns=MODELS, index=['accuracy', 'precision', 'recall', 'f1_score'])

    for model in MODELS:
        df = raw_results[model]
        y_true = df['y_true_idx']
        y_pred = df['y_pred_idx']

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_excel(writer, sheet_name=model)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        synthesized_results[model] = [acc, prec, rec, f1]

    synthesized_results.to_excel(writer, sheet_name='synthesized_metrics')


# ============================================================================
# COCHRAN Q TEST AND POST-HOC ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("COCHRAN Q TEST: Comparing Model Performance Across Treatments")
print("="*70)

args = [ raw_results[model]['y_pred_idx'].to_numpy() for model in MODELS ]
args.insert(0, raw_results[MODELS[0]]['y_true_idx'].to_numpy())  # Add y_true as the first argument
q_stat, p_value = cochrans_q(*args)

print(f"Cochran's Q Statistic: {q_stat:.4f}")
print(f"P-value: {p_value:.4e}")


if p_value < 0.05:
    print("\nThe Cochran's Q test indicates significant differences among the models. Proceeding with post-hoc analysis...")
    table_df = pd.DataFrame(index=MODELS, columns=MODELS)
    table_df.fillna('-', inplace=True)

    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            table = mcnemar_table(raw_results[MODELS[i]]['y_true_idx'], raw_results[MODELS[i]]['y_pred_idx'],
                                  raw_results[MODELS[j]]['y_pred_idx'])
            chi2, p_value = mcnemar(table, corrected=False)

            if p_value < 0.05:
                print(f"\nSignificant difference between {MODELS[i]} vs {MODELS[j]}:")
                print(f'Accuracy {MODELS[i]}: {accuracy_score(raw_results[MODELS[i]]["y_true_idx"], raw_results[MODELS[i]]["y_pred_idx"])}')
                print(f'Accuracy {MODELS[j]}: {accuracy_score(raw_results[MODELS[j]]["y_true_idx"], raw_results[MODELS[j]]["y_pred_idx"])}')
                print(f"McNemar's Test Statistic: {chi2}")
                print(f"P-value: {p_value}")

                table_df.loc[MODELS[i], MODELS[j]] = p_value


print(table_df)

with pd.ExcelWriter(f'{OUTPUT_FOLDER}/posthoc_analysis.xlsx') as writer:
    table_df.to_excel(writer, sheet_name='mcnemar_pvalues')

print("="*70 + "\n")

sys.exit(0)
