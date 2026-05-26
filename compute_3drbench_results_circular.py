import os

import numpy as np
import pandas as pd

################
dataset_name = '3DSRBenchv1'
results_path = 'outputs/3dsrbench'
results_file = f'results_{dataset_name}.csv'
################

LABELS = ['A', 'B', 'C', 'D']

mapping = {
    'orientation': ['orientation_in_front_of', 'orientation_on_the_left'],
}

types = ['orientation']

subtypes = sum([mapping[k] for k in types], [])

file_mapping = {}
for model in os.listdir(results_path):
    file = os.path.join(results_path, model, f'{model}_{dataset_name}_openai_result.xlsx')
    if os.path.isfile(file):
        file_mapping[model] = file

# Compute model results
results_csv = []
for model in file_mapping:
    file = file_mapping[model]
    df = pd.read_excel(file)

    results = {}
    for i in range(len(df.index)):
        row = df.iloc[i].tolist()

        assert row[12] in [0, 1], row

        if row[1][-2] == '-':
            qid = row[1][:-2]
        else:
            qid = row[1]

        if qid in results:
            results[qid][0] = results[qid][0] * row[12]
        else:
            results[qid] = [row[12], row[8]]

        assert row[8] in subtypes, row[8]

    curr_results = [np.mean([results[k][0] for k in results])]
    # print(len([results[k][0] for k in results]))
    for t in types:
        # print(t, len([results[k][0] for k in results if results[k][1] in mapping[t]]))
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] in mapping[t]]))
    for t in subtypes:
        curr_results.append(np.mean([results[k][0] for k in results if results[k][1] == t]))
    # exit()

    curr_results = [model] + [np.round(v*100, decimals=2) for v in curr_results]

    results_csv.append(curr_results)

# Compute a random baseline
file = file_mapping[model]
df = pd.read_excel(file)
results = {}
for i in range(len(df.index)):
    row = df.iloc[i].tolist()
    assert row[12] in [0, 1], row
    if row[1][-2] == '-':
        qid = row[1][:-2]
    else:
        qid = row[1]
    if isinstance(row[4], float):
        hit = int(np.random.randint(2) == 0)
    else:
        hit = int(np.random.randint(4) == 0)
    if qid in results:
        results[qid][0] = results[qid][0] * hit
    else:
        results[qid] = [hit, row[8]]
    assert row[8] in subtypes, row[8]
curr_results = [np.mean([results[k][0] for k in results])]
for t in types:
    curr_results.append(np.mean([results[k][0] for k in results if results[k][1] in mapping[t]]))
for t in subtypes:
    curr_results.append(np.mean([results[k][0] for k in results if results[k][1] == t]))
curr_results = ['random'] + [np.round(v*100, decimals=2) for v in curr_results]
results_csv.append(curr_results)

pd.DataFrame(columns=['model', 'overall']+types+subtypes, data=results_csv).to_csv(results_file, index=False)
