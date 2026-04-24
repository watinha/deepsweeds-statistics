import pandas as pd, os, sys

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


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

sys.exit(0)
