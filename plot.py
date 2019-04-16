import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

main_directory = '/clusterhome/chaudhry/experiments'
best_Performers = pd.DataFrame()

# Find Performers from Every Architecture
for number_of_Nodes in range(1,11):
    for middle_layer in range(100,101): 
        architecture = '{:02}x{:02}x{:02}'.format(number_of_Nodes, middle_layer * number_of_Nodes, number_of_Nodes)
        eval_df = pd.read_csv("{}/evaluation/set12/{}.csv".format(main_directory, architecture), delim_whitespace=True)
        specific_df = eval_df.loc[eval_df['Architecture'] == architecture]
        specific_df
        selected_df = specific_df.sort_values('MSE')
        print selected_df[['Name', 'MSE', 'Validation_MSE']][0:1]
        best_Performers = best_Performers.append(selected_df[0:1], ignore_index=True)

best_Performers[['Name', 'MSE', 'Validation_MSE']]

# Plot Best Performers for every Architecture
plt.figure(1)
for i in range(10):
    plt.plot([1,2,3,4,5,6,7,8,9,10], list(best_Performers['MSE'][[0+i, 10+i, 20+i, 30+i, 40+i, 50+i, 60+i, 70+i, 80+i, 90+i]]), 'x')
plt.xlabel('Number of Neurons')
plt.xticks(np.arange(1, 11, 1.0))
plt.ylabel('L2 Distance from Target Muscle Activation')
plt.legend(['N x N x N', 'N x 2N x N', 'N x 3N x N', 'N x 4N x N', 'N x 5N x N', 'N x 6N x N', 'N x 7N x N', 'N x 8N x N', 'N x 9N x N', 'N x 10N x N'], title = 'Internal Architecture', loc='upper right')
plt.savefig('/clusterhome/chaudhry/experiments/plots/Set12_Performers_.png', )

# Plot Best Performers for every Architecture Excluding 1
plt.figure(2)
for i in range(2):
    plt.plot([2,3,4,5,6,7,8,9,10], list(best_Performers['MSE'][[10+i, 20+i, 30+i, 40+i, 50+i, 60+i, 70+i, 80+i, 90+i]]), 'x')
plt.xlabel('Number of Neurons')
plt.xticks(np.arange(1, 11, 1.0))
plt.ylabel('L2 Distance from Target Muscle Activation')
plt.legend(['N x N x N', 'N x 2N x N', 'N x 3N x N', 'N x 4N x N', 'N x 5N x N', 'N x 6N x N', 'N x 7N x N', 'N x 8N x N', 'N x 9N x N', 'N x 10N x N'], title = 'Internal Architecture', loc='upper right')
plt.savefig('/clusterhome/chaudhry/experiments/plots/Set12_Performers_reduced.png', )

plt.show()


# pdf = matplotlib.backends.backend_pdf.PdfPages('/clusterhome/chaudhry/experiments/plots/Set3_Performers.pdf')
# for fig in xrange(1, plt.gcf().number+1):
#     pdf.savefig( fig )
# pdf.close()

# # Best Performers for every External Node Count
# for number_of_Nodes in range(1, 111):
#     architecture = '{:02}x{:02}x{:02}'.format(number_of_Nodes, 10 * number_of_Nodes, number_of_Nodes)
#     eval_df = pd.read_csv("{}/evaluation/set12/{}.csv".format(main_directory, architecture), delim_whitespace=True)
#     selected_df = eval_df.sort_values('MSE')
#     print selected_df[['Name', 'MSE', 'Validation_MSE']][0:1]
#     best_Performers = best_Performers.append(selected_df[0:1], ignore_index=True)
