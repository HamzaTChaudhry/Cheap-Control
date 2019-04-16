from network import construct, evaluate
import pandas as pd
import datetime
import sys 

def main():
    date = str(datetime.datetime.today().strftime('%d-%m-%Y'))

    main_directory = "/clusterhome/chaudhry/experiments"
    update_rate = 10
    input_data = "trial_1_halfres_convergent_" + str(update_rate)
    set_number = 12

    number_of_Nodes    = int(sys.argv[1])

    evaluation_df = pd.DataFrame(columns = ['Name', 'Architecture', 'Outer_Layer', 'Middle_Layer', 'Model_Index', 'MSE', 'Validation_MSE', 'R^2', 'Explained_Variance', 'Predicted_Value', 'Weights', 'Date'])

    for middle_layer in 10 * range(10, 11):
        for model_index in range(50):
            architecture = "{:02}x{:03}x{:02}".format(number_of_Nodes, 10 * middle_layer, number_of_Nodes)
            name = "FF_{}-{:02}".format(architecture, model_index)

            # Print Network Information
            print "Network Architecture: {}. Model Index: {:02}".format(architecture, model_index)

            # Create Network and Evaluate Metrics of Network
            network = construct(main_directory, input_data, set_number, number_of_Nodes, middle_layer, model_index)
            MSE, validation_MSE, r2, explained_var, prediction, weights = evaluate(main_directory, input_data, network)
            
            # Add This Network and its Evaluation to a Dataframe
            evaluation_data = pd.Series([name, architecture, number_of_Nodes, 10 * middle_layer, model_index, MSE, validation_MSE, r2, explained_var, prediction, weights, date], index = evaluation_df.columns)
            
            evaluation_df = evaluation_df.append([evaluation_data], ignore_index=True)
            
            # Save the Evaluation Dataframe after Each Model
            evaluation_df.to_csv("{}/evaluation/set{}/{}.csv".format(main_directory, set_number, architecture), sep='\t', index=False)    


if __name__ == "__main__":
    main()

