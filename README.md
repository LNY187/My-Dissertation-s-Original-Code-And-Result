# Analysis of Socioeconomic Factors and Crime Rates in China

This project investigates the relationship between socioeconomic development and crime rates across China's provinces from 2013 to 2019. It uses a multi-stage data processing pipeline in Python and a comprehensive statistical analysis in R to explore these complex dynamics.

## Project Structure

The project is organized into two main parts:

-   `Data Cleaning/`: Contains the Python pipeline for cleaning, standardizing, and merging various raw datasets.
-   `Data Analysis/`: Contains the R scripts for statistical analysis, regression modeling, and visualization.

## Data

**Note:** The `ChinaCrimData.csv` file is not included in this repository due to its large size. It can be downloaded from the following source:

-   [A decade of crime in China](https://www.nature.com/articles/s41597-025-04757-8)

## Data Cleaning

The data cleaning process is managed by a series of Python scripts, orchestrated by a master pipeline script.

### Pipeline Phases

1.  **Phase 1: Geographic Standardization**: Standardizes provincial and city names.
2.  **Phase 2: Crime Data Processing**: Aggregates and processes crime data.
3.  **Phase 3: Socioeconomic Data Processing**: Merges and cleans various socioeconomic indicators.
4.  **Phase 4: Final Dataset Creation**: Creates the final panel dataset for analysis.

### How to Run the Pipeline

To run the entire data cleaning pipeline, execute the `master_pipeline.py` script:

```bash
python "Data Cleaning/master_pipeline.py"
```

You can also run individual phases by passing the phase number as an argument:

```bash
python "Data Cleaning/master_pipeline.py" 1
```

### Dependencies

The Python scripts require the following libraries:

-   `pandas`
-   `numpy`
-   `openpyxl`
-   `xlrd`

These dependencies are listed in the `Data Cleaning/.venv` directory and can be installed using `pip`.

## Data Analysis

The statistical analysis is conducted using R scripts that perform regression analysis, correlation tests, and other statistical validations.

### Analysis Scripts

-   `question1_academic_analysis.R`: Focuses on the relationship between GDP per capita and crime rates, including panel data models.
-   `enhanced_regression_analysis.R`: Provides a more comprehensive analysis with multicollinearity checks, trend variables, and robustness tests.

### How to Run the Analysis

To run the R scripts, you will need to have R and RStudio installed. Open the scripts in RStudio and execute them. Make sure to set the working directory to the `Data Analysis` folder.

### Dependencies

The R scripts require the following libraries:

-   `readr`
-   `dplyr`
-   `ggplot2`
-   `corrplot`
-   `plm`
-   `stargazer`
-   `quantreg`
-   `car`
-   `lmtest`
-   `sandwich`

You can install these packages in R using `install.packages("package_name")`.

## Results

The analysis generates several outputs, including:

-   **Correlation plots**: Visualizing the relationships between different variables.
-   **Regression model summaries**: Saved as `.rds` files in the `Data Analysis` directory.
-   **Final panel dataset**: `complete_panel_dataset.csv` in the `Data Analysis/data` directory.

The key findings suggest a positive correlation between economic development and crime rates, with significant regional variations.


