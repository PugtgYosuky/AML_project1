# AML_project1
IDS detection

## Packages versions
### Python
- Python: version 3.8.15

### Libraries
The requirements.txt file contains all the version of the libraries used in this program. 

To run the program, it is highly recommended to use the these versions.


## RUN program

### Download datasets
First, you must put the training and test datasets in the directory src/dataset
The data can be downloaded from https://www.kaggle.com/competitions/aml-2023-projet1/data-

To install the recommended, the following command should be run in the terminal:
<code> pip install -r requirements.txt </code>

### Command 
To use the program, the following command should be run in the terminal:
<code> 
cd src
& 
python main.py config.json

</code>

The *config.json* file contains all the information needed to test different parameters. His content in described in the subsection bellow.

### Configuration file
It shold be given to the program a json file with the following parameters:

<table>

<tr>
    <th> Parameter </th>
    <th> Description </th>
    <th> Available options </th>
    <th> Mandatory </th>
    <th> Default </th>
    <th> Data type </th>
</tr>

<tr>
    <td> dataset </td>
    <td> Path to the dataset </td>
    <td> - </td>
    <td> - </td>
    <td> Yes </td>
    <td> String </td>
</tr>

<tr>
    <td> target_dataset </td>
    <td> Path to the taget dataset (for the Kaggle competition) </td>
    <td> - </td>
    <td> - </td>
    <td> Yes </td>
    <td> String </td>
</tr>

<tr>
    <td> norm_model </td>
    <td> Normalization / standarization method </td>
    <td> 'MinMax' (MinMaxScaler), 'Robust' (RobustScaler), 'StandardScaler' </td>
    <td> StandardScaler </td>
    <td> No </td>
    <td> String </td>
</tr>

<tr>
    <td> balance_dataset </td>
    <td> The method to balance the classes </td>
    <td> 'SMOTE', 'SMOTETomek', 'SMOTEENN', null </td>
    <td> null </td>
    <td> No </td>
    <td> String </td>
</tr>

<tr>
    <td> columns_to_drop </td>
    <td> List with the columns of the dataset to drop </td>
    <td> - </td>
    <td> [] </td>
    <td> No </td>
    <td> List of strings </td>
</tr>

<tr>
    <td> number_best_features </td>
    <td> Number of best features to use according to the SelectKBest model from sklearn </td>
    <td> - </td>
    <td> 'all' </td>
    <td> No </td>
    <td> integer os 'all' </td>
</tr>

<tr>
    <td> variance_threshold </td>
    <td> Minimum threshold of variance for the VarianceThreshold model from sklearn</td>
    <td> - </td>
    <td> 0.0 </td>
    <td> No </td>
    <td> float </td>
</tr>

<tr>
    <td> num_best_models </td>
    <td> Number of best models of each type that shoulb be selected wo predict the test dataset</td>
    <td> - </td>
    <td> 3 </td>
    <td> No </td>
    <td> integer </td>
</tr>

<tr>
    <td> models_names </td>
    <td> List with the models and parameters to use for each model. In case of grid search, each parameter should be a list with the values to test</td>
    <td> - </td>
    <td> Example usage: [[{model's name}, {dict with the parameters}]] </td>
    <td> Yes </td>
    <td> list of list </td>
</tr>

<tr>
    <td> grid_search </td>
    <td> Boolean to use grid search for the best settings. It uses cross validation to dataset to search for the best configs.</td>
    <td> - </td>
    <td> false </td>
    <td> No </td>
    <td> boolean </td>
</tr>

<tr>
    <td> train_only </td>
    <td> Boolean to only train the given models and parameters with the entire dataset and predict the test dataset</td>
    <td> - </td>
    <td> false </td>
    <td> No </td>
    <td> boolean </td>
</tr>

num_best_models

</table>
