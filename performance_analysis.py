import pandas as pd, os, sys, numpy as np


from scipy.stats import shapiro, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman


PERFORMANCE_FILES = {
    'low': 'benchmark_details_device_slow.csv',
    'medium': 'benchmark_details_medium.csv',
    'high': 'benchmark_details_high.csv'
}

MODELS = [ 'efficientnetv2b0', 'efficientnetv2b1', 'efficientnetv2b2', 'efficientnetv2b3',
           'inceptionv3', 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large', 'nasnetmobile', 
           'resnet101v2', 'resnet50' ]


dev_dfs = {}
dev_model_dfs = {}
for device, filename in PERFORMANCE_FILES.items():
    dev_dfs[device] = pd.read_csv(f'./performance/{filename}')

    model_name_list = [ filename.split('_')[1] for filename in dev_dfs[device]['nome_arquivo'] ]
    dev_dfs[device]['model'] = model_name_list

    for model in MODELS:
        group_id = f'{device}_{model}'
        dev_model_dfs[group_id] = dev_dfs[device].loc[dev_dfs[device]['model'] == model]


def run_analysis_on (metric, dfs):
    print(f"\n{'='*70}\nAnalyzing {metric}...\n{'='*70}")

    summary_df = pd.DataFrame(columns=['group_id', 'mean', 'std', 'min', 'max'])
    group_ids = []
    means = []
    stds = []
    mins = []
    maxs = []

    for group_id, df in dfs.items():
        group_ids.append(group_id)
        means.append(df[metric].mean())
        stds.append(df[metric].std())
        mins.append(df[metric].min())
        maxs.append(df[metric].max())

    summary_df['group_id'] = group_ids
    summary_df['mean'] = means
    summary_df['std'] = stds
    summary_df['min'] = mins
    summary_df['max'] = maxs


    shapiro_results = pd.DataFrame(columns=['group_id', 'statistic', 'p_value'])
    group_ids = []
    statistics = []
    p_values = []
    parametric = True

    for group_id, df in dfs.items():
        stat, p_value = shapiro(df[metric])
        group_ids.append(group_id)
        statistics.append(stat)
        p_values.append(p_value)

        if p_value < 0.05:
            parametric = False


    if parametric:
        print("All groups are normally distributed. Proceeding with parametric tests.")

    else:
        print("At least one group is not normally distributed. Consider using non-parametric tests.")
        statistic, p_value = friedmanchisquare(*[df[metric] for df in dfs.values()])
        group_ids.append('friedman_test')
        statistics.append(statistic)
        p_values.append(p_value)
        print(f"Friedman test statistic: {statistic:.4f}, p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Significant differences found among groups. Performing post-hoc Nemenyi test.")
            
            metric_array = np.array([df[metric].to_numpy() for df in dfs.values()])
            metric_array = metric_array.T
            posthoc_results = posthoc_nemenyi_friedman(metric_array)
            posthoc_results.index = dfs.keys()
            posthoc_results.columns = dfs.keys()


    shapiro_results['group_id'] = group_ids
    shapiro_results['statistic'] = statistics
    shapiro_results['p_value'] = p_values


    with pd.ExcelWriter(f'./resultados/performance_analysis_{metric}.xlsx') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        shapiro_results.to_excel(writer, sheet_name='shapiro_results', index=False)
        posthoc_results.to_excel(writer, sheet_name='posthoc_nemenyi', index=True)


if __name__ == "__main__":
    run_analysis_on('inference_time', dev_model_dfs)
    run_analysis_on('pss_peak', dev_model_dfs)






