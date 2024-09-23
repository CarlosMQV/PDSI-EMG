# Ninapro Dataset (DB6)

This folder contains files from the **Ninapro** dataset. You can find more information about the dataset at the following link: [Ninapro DB6](https://ninapro.hevs.ch/instructions/DB6.html).

### File Naming Convention

The `.parquet` file names are coded as follows:

"S1_D1_T1.parquet"

- **S1**: Subject 1
- **D1**: Day 1
- **T1**: Time of day 1 (morning; whereas **T2** represents afternoon)

Each `.parquet` file contains synchronized variables for each exercise and subject.

## Dataset Variables

The variables included in the `.parquet` files are as follows:

- **emg** (16 columns): sEMG signals from the 14 electrodes (2 columns are empty)
- **stimulus** (1 column): the movement repeated by the subject

## Loading Data with `pandas.read_parquet` as a `DataFrame`

To load the data from the `.parquet` files in Python, you can use the `read_parquet` function from the `pandas` package. Here is an example of how to do it.

### Example Usage with `read_parquet`

```python
import pandas as pd

# Load the .parquet file
data = pd.read_parquet('path_to_file/S1_D1_T1.parquet')

# General Summary
print("General Summary:")
df.info()

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())
```