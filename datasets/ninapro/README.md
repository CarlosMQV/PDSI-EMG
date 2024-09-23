# Ninapro Dataset (DB6)

This folder contains files from the **Ninapro** dataset. You can find more information about the dataset at the following link: [Ninapro DB6](https://ninapro.hevs.ch/instructions/DB6.html).

### File Naming Convention

The `.mat` file names are coded as follows:

"S1_D1_T1.mat"

- **S1**: Subject 1
- **D1**: Day 1
- **T1**: Time of day 1 (morning; whereas **T2** represents afternoon)

Each `.mat` file contains synchronized variables for each exercise and subject.

## Dataset Variables

The variables included in the `.mat` files are as follows:

- **subj**: subject number
- **acc** (48 columns): three-axis accelerometers for the 12 electrodes (6 columns are empty)
- **emg** (16 columns): sEMG signals from the 14 electrodes (2 columns are empty)
- **stimulus** (1 column): the movement repeated by the subject
- **restimulus** (1 column): the movement repeated by the subject, but in this case, the movement label duration is refined a-posteriori to match the real movement
- **object** (1 column): the object used
- **reobject** (1 column): relabeled used object
- **repetition** (1 column): repetition of the stimulus
- **rerepetition** (1 column): repetition of the restimulus
- **repetition object** (1 column): repetition of the used object
- **daytesting**: day of the acquisition (1 to 5)
- **time**: time of the acquisition (1 for morning, 2 for afternoon)

## Loading Data with `scipy.io.loadmat`

To load the data from the `.mat` files in Python, you can use the `loadmat` function from the `scipy.io` package. Here is an example of how to do it.

### Example Usage with `loadmat`

```python
from scipy.io import loadmat

# Load the .mat file
data = loadmat('path_to_file/S1_D1_T1.mat')

# Access the variables
emg_data = data['emg']  # sEMG signal
acc_data = data['acc']  # Accelerometer
```

### `loadmat` Description

The `scipy.io.loadmat` function loads MATLAB (.mat) files into Python. It returns a dictionary where the keys are the variable names in the `.mat` file and the values are the corresponding data.

- **Key Parameters**:
  - `file_name`: Path to the `.mat` file to be loaded.
  - `simplify_cells` (optional): If `True`, it simplifies MATLAB cell elements into Python lists or structures.

For more details, see the official SciPy documentation: [scipy.io.loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).

## Converting Data to a Pandas `DataFrame`

Once the data is loaded using `loadmat`, it is recommended to convert it into a more manageable format like a **pandas** `DataFrame`. `DataFrames` allow you to work with the data in a structured way and perform analysis more easily.

### Example Conversion to `DataFrame`

```python
import pandas as pd
from scipy.io import loadmat

# Load the .mat file
data = loadmat('path_to_file/S1_D1_T1.mat')

# Convert the sEMG signal to a DataFrame
emg_df = pd.DataFrame(data['emg'], columns=[f'emg_{i+1}' for i in range(data['emg'].shape[1])])

# Convert the accelerometer data to a DataFrame
acc_df = pd.DataFrame(data['acc'], columns=[f'acc_{i+1}' for i in range(data['acc'].shape[1])])

# Display the first records
print(emg_df.head())
print(acc_df.head())
```