# Question 1: Provincial Level Analysis of GDP Per Capita and Crime Rates

# Academic Research Framework with Theoretical Foundation

library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(plm)
library(stargazer)
library(quantreg)

# Load and prepare data
# Set working directory to current workspace
setwd("/Users/harmonalex/Desktop/0045_revised_R_code")
data <- read_csv("data/complete_panel_dataset.csv")

# Apply population unit conversion
data <- data %>%
  mutate(
    needs_correction = ifelse(!is.na(Total_Crime_Rate) & Total_Crime_Rate > 50, TRUE, FALSE),
    Total_Crime_Rate = ifelse(needs_correction,
                              (Total_Crimes / (Total_Population * 10000)) * 100000,
                              Total_Crime_Rate),
    Violent_Crime_Rate = ifelse(needs_correction,
                                (Violent_Crimes / (Total_Population * 10000)) * 100000,
                                Violent_Crime_Rate),
    Property_Crime_Rate = ifelse(needs_correction,
                                 (Property_Crimes / (Total_Population * 10000)) * 100000,
                                 Property_Crime_Rate)
  ) %>%
  select(-needs_correction)

# Prepare analysis dataset with theoretical framework
analysis_data <- data %>%
  filter(!is.na(Total_Crime_Rate) & !is.na(GDP_Per_Capita) & 
         !is.na(Violent_Crime_Rate) & !is.na(Property_Crime_Rate)) %>%
  mutate(
    # Theoretical variables based on criminological theories
    Economic_Opportunity_Index = GDP_Per_Capita * Urbanization_Rate,
    Relative_Deprivation_Index = Urban_Rural_Income_Ratio * (1 - Urbanization_Rate/100),
    Development_Stage = case_when(
      GDP_Per_Capita < 30000 ~ "Early_Development",
      GDP_Per_Capita < 60000 ~ "Middle_Development", 
      TRUE ~ "Advanced_Development"
    ),
    GDP_Per_Capita_Squared = GDP_Per_Capita^2,
    Time_Trend = Year - 2012,
    Period_Dummy = ifelse(Year > 2015, 1, 0)
  )

cat("=== QUESTION 1: ACADEMIC ANALYSIS ===\n")
cat("Observations:", nrow(analysis_data), "\n")
cat("Provinces:", length(unique(analysis_data$Province)), "\n")
cat("Years:", min(analysis_data$Year), "-", max(analysis_data$Year), "\n\n")

# 1. Theoretical correlation analysis
theoretical_correlations <- analysis_data %>%
  group_by(Year) %>%
  summarise(
    GDP_Total_Crime_Cor = cor(GDP_Per_Capita, Total_Crime_Rate, use = "complete.obs"),
    GDP_Violent_Crime_Cor = cor(GDP_Per_Capita, Violent_Crime_Rate, use = "complete.obs"),
    GDP_Property_Crime_Cor = cor(GDP_Per_Capita, Property_Crime_Rate, use = "complete.obs"),
    Opportunity_Total_Crime_Cor = cor(Economic_Opportunity_Index, Total_Crime_Rate, use = "complete.obs"),
    Deprivation_Total_Crime_Cor = cor(Relative_Deprivation_Index, Total_Crime_Rate, use = "complete.obs"),
    .groups = 'drop'
  )

cat("Theoretical Variable Correlations by Year:\n")
print(round(theoretical_correlations, 3))

# 2. Causal identification with panel data
panel_data <- pdata.frame(analysis_data, index = c("Province", "Year"))

# Fixed effects models
model1 <- plm(Total_Crime_Rate ~ GDP_Per_Capita, data = panel_data, model = "within")
model2 <- plm(Total_Crime_Rate ~ GDP_Per_Capita + factor(Year), data = panel_data, model = "within")
model3 <- plm(Total_Crime_Rate ~ Economic_Opportunity_Index + Relative_Deprivation_Index + 
              factor(Year), data = panel_data, model = "within")

# Center GDP_Per_Capita to reduce multicollinearity
analysis_data$GDP_Per_Capita_Centered <- scale(analysis_data$GDP_Per_Capita, center = TRUE, scale = FALSE)[,1]
analysis_data$GDP_Per_Capita_Squared_Centered <- analysis_data$GDP_Per_Capita_Centered^2

# Update panel data
panel_data <- pdata.frame(analysis_data, index = c("Province", "Year"))

model4 <- plm(Total_Crime_Rate ~ GDP_Per_Capita_Centered + GDP_Per_Capita_Squared_Centered + 
              factor(Year), data = panel_data, model = "within")
model5 <- plm(Total_Crime_Rate ~ GDP_Per_Capita * factor(Development_Stage) + 
              factor(Year), data = panel_data, model = "within")

cat("\nPanel Data Models:\n")
cat("Model 1: Basic Fixed Effects\n")
print(summary(model1))
cat("\nModel 2: With Time Fixed Effects\n")
print(summary(model2))
cat("\nModel 3: Theoretical Variables\n")
print(summary(model3))
cat("\nModel 4: Nonlinear Relationship (Centered)\n")
tryCatch({
  print(summary(model4))
}, error = function(e) {
  cat("Error in Model 4:", e$message, "\n")
  cat("Skipping nonlinear model due to multicollinearity\n")
})
cat("\nModel 5: Development Stage Interactions\n")
tryCatch({
  print(summary(model5))
}, error = function(e) {
  cat("Error in Model 5:", e$message, "\n")
  cat("Skipping interaction model due to insufficient variation\n")
})

# 3. Robustness tests
early_period <- analysis_data %>% filter(Year <= 2015)
late_period <- analysis_data %>% filter(Year > 2015)
early_cor <- cor(early_period$GDP_Per_Capita, early_period$Total_Crime_Rate, use = "complete.obs")
late_cor <- cor(late_period$GDP_Per_Capita, late_period$Total_Crime_Rate, use = "complete.obs")

cat("\nRobustness Test Results:\n")
cat("Early Period (2013-2015) Correlation:", round(early_cor, 3), "\n")
cat("Late Period (2016-2019) Correlation:", round(late_cor, 3), "\n")
cat("Correlation Change:", round(late_cor - early_cor, 3), "\n")

# 4. Development stage analysis
development_analysis <- analysis_data %>%
  group_by(Development_Stage, Year) %>%
  summarise(
    N_Provinces = n(),
    Mean_GDP_Per_Capita = mean(GDP_Per_Capita, na.rm = TRUE),
    Mean_Total_Crime_Rate = mean(Total_Crime_Rate, na.rm = TRUE),
    GDP_Crime_Correlation = cor(GDP_Per_Capita, Total_Crime_Rate, use = "complete.obs"),
    .groups = 'drop'
  )

cat("\nDevelopment Stage Analysis:\n")
print(development_analysis)

# 5. Key findings
overall_correlation <- cor(analysis_data$GDP_Per_Capita, analysis_data$Total_Crime_Rate, use = "complete.obs")
high_dev_crime <- mean(analysis_data$Total_Crime_Rate[analysis_data$Development_Stage == "Advanced_Development"], na.rm = TRUE)
low_dev_crime <- mean(analysis_data$Total_Crime_Rate[analysis_data$Development_Stage == "Early_Development"], na.rm = TRUE)

cat("\nKey Academic Findings:\n")
cat("1. Overall GDP-Crime Correlation:", round(overall_correlation, 3), "\n")
cat("2. Early vs Late Period Change:", round(late_cor - early_cor, 3), "\n")
cat("3. Advanced Development Crime Rate:", round(high_dev_crime, 2), "per 100,000\n")
cat("4. Early Development Crime Rate:", round(low_dev_crime, 2), "per 100,000\n")
cat("5. Development Gap:", round(high_dev_crime - low_dev_crime, 2), "crimes per 100,000\n")

cat("\nTheoretical Implications:\n")
cat("1. Opportunity Theory: Economic development creates crime opportunities\n")
cat("2. Relative Deprivation: Income inequality affects crime patterns\n")
cat("3. Development Stages: Different stages show different crime patterns\n")

# Save results
saveRDS(list(
  theoretical_correlations = theoretical_correlations,
  panel_models = list(model1, model2, model3, model4, model5),
  development_analysis = development_analysis,
  key_findings = list(
    overall_correlation = overall_correlation,
    early_late_comparison = list(early = early_cor, late = late_cor),
    development_gap = high_dev_crime - low_dev_crime
  )
), "question1_academic_results.rds")

cat("\n=== ACADEMIC ANALYSIS COMPLETE ===\n")
cat("Results saved to: question1_academic_results.rds\n")
