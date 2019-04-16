import pandas as pd

main_directory = '/home/hamza/Research'
best_Performers = pd.DataFrame()
set_number = 12

# Find Performers from Every Architecture
for number_of_Nodes in range(1,8):
    for middle_layer in range(1,11): 
        architecture = '{:02}x{:03}x{:02}'.format(number_of_Nodes, middle_layer * 10, number_of_Nodes)
        eval_df = pd.read_csv("/clusterhome/chaudhry/experiments/evaluation/set{}/{}.csv".format(set_number, architecture), delim_whitespace=True)
        specific_df = eval_df.loc[eval_df['Architecture'] == architecture]
        specific_df
        selected_df = specific_df.sort_values('MSE')
        print selected_df[['Name', 'MSE', 'Validation_MSE']][1:2]
        best_Performers = best_Performers.append(selected_df[1:2], ignore_index=True)

saved_df = best_Performers[['Name', 'MSE', 'Validation_MSE']].iloc[:, 0:4]
saved_df.to_csv("clusterhome/chaudhry/experiments/evaluation/set{}/best_performers.csv".format(set_number), sep='\t', index=False)