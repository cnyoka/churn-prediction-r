# ============================================================
# Capstone A1 – Customer Churn Prediction
# UK Online Gift Retailer | UCI Online Retail II
# ============================================================

# Packages
install.packages(c("tidyverse", "lubridate", "caret", "pROC", "rpart", "rpart.plot"))
library(tidyverse)
library(lubridate)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)

# ============================================================
# Load & clean data
# ============================================================
raw <-  read.csv(file.choose())

# Revenue calculation, removing returns/bad data and Parsing Date
clean <- raw %>%
  mutate(
    InvoiceDate = as.POSIXct(InvoiceDate, format = "%Y-%m-%d %H:%M:%S"),
    Revenue = Quantity * Price
  ) %>%
  filter(
    Quantity > 0,
    Price > 0,
    !is.na(Customer.ID)      # must have a customer ID
  )

cat("Clean rows:", nrow(clean), "\n")          
cat("Unique customers:", n_distinct(clean$Customer.ID), "\n")  

# ============================================================
# Definitions: observation and outcome windows
# ============================================================
obs_start  <- as.POSIXct("2009-12-01")
obs_end    <- as.POSIXct("2011-06-30")
out_start  <- as.POSIXct("2011-07-01")
out_end    <- as.POSIXct("2011-11-30")

obs  <- clean %>% filter(InvoiceDate >= obs_start & InvoiceDate <= obs_end)
outcome <- clean %>% filter(InvoiceDate >= out_start & InvoiceDate <= out_end)

cat("Customers in observation window:", n_distinct(obs$Customer.ID), "\n")   
cat("Customers in outcome window:", n_distinct(outcome$Customer.ID), "\n")  

# ============================================================
# RFM features from observation window
# ============================================================
reference_date <- as.POSIXct("2011-07-01")

rfm <- obs %>%
  group_by(Customer.ID) %>%
  summarise(
    recency   = as.numeric(difftime(reference_date, max(InvoiceDate), units = "days")),
    frequency = n_distinct(Invoice),
    monetary  = sum(Revenue, na.rm = TRUE),
    tenure    = as.numeric(difftime(reference_date, min(InvoiceDate), units = "days"))
  ) %>%
  ungroup()

# ============================================================
# Churn labels
# ============================================================
retained_customers <- outcome %>% distinct(Customer.ID) %>% mutate(retained = 1)

model_data <- rfm %>%
  left_join(retained_customers, by = "Customer.ID") %>%
  mutate(
    retained = replace_na(retained, 0),
    churned  = 1 - retained       # 1 = churned, 0 = retained
  )

cat("Total customers in model:", nrow(model_data), "\n")   
cat("Churned:", sum(model_data$churned), "\n")             
cat("Retained:", sum(model_data$retained), "\n")           
cat("Churn rate:", round(mean(model_data$churned) * 100, 1), "%\n") 

# ============================================================
# Exploratory plots
# ============================================================

# Plot 1: Churn rate bar chart
model_data %>%
  count(churned) %>%
  mutate(label = ifelse(churned == 1, "Churned", "Retained")) %>%
  ggplot(aes(x = label, y = n, fill = label)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = n), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("Churned" = "#e07b39", "Retained" = "#3266ad")) +
  labs(title = "Customer Churn vs Retention",
       subtitle = "UK Online Gift Retailer | Outcome window: Jul–Nov 2011",
       x = NULL, y = "Number of Customers") +
  theme_minimal() +
  theme(legend.position = "none")


# Plot 2: Recency distribution by churn status
ggplot(model_data, aes(x = recency, fill = factor(churned))) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("0" = "#3266ad", "1" = "#e07b39"),
                    labels = c("Retained", "Churned")) +
  labs(title = "Recency Distribution by Churn Status",
       subtitle = "Churned customers had their last purchase further in the past",
       x = "Days Since Last Purchase", y = "Count", fill = NULL) +
  theme_minimal()


# Plot 3: Boxplot of frequency by churn
ggplot(model_data, aes(x = factor(churned), y = frequency,
                       fill = factor(churned))) +
  geom_boxplot(outlier.shape = NA) +
  scale_fill_manual(values = c("0" = "#3266ad", "1" = "#e07b39")) +
  scale_x_discrete(labels = c("Retained", "Churned")) +
  coord_cartesian(ylim = c(0, 30)) +
  labs(title = "Purchase Frequency by Churn Status",
       subtitle = "Retained customers purchased more frequently",
       x = NULL, y = "Number of Orders", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "none")

# ============================================================
# Train / test split
# ============================================================
set.seed(42)

model_input <- model_data %>%
  select(recency, frequency, monetary, tenure, churned) %>%
  mutate(churned = factor(churned, levels = c(0, 1),
                          labels = c("Retained", "Churned")))

train_index <- createDataPartition(model_input$churned, p = 0.75, list = FALSE)
train <- model_input[train_index, ]
test  <- model_input[-train_index, ]

cat("Training rows:", nrow(train), "\n")  
cat("Test rows:", nrow(test), "\n")        

# ============================================================
# Logistic regression model
# ============================================================
logit_model <- glm(churned ~ recency + frequency + monetary + tenure,
                   data = train %>% mutate(churned = ifelse(churned == "Churned", 1, 0)),
                   family = binomial)

summary(logit_model) 

# Predict on test set
logit_probs <- predict(logit_model,
                       newdata = test %>% mutate(churned = ifelse(churned == "Churned", 1, 0)),
                       type = "response")
logit_pred  <- ifelse(logit_probs > 0.5, "Churned", "Retained")
logit_pred  <- factor(logit_pred, levels = c("Retained", "Churned"))

# Confusion matrix
cm_logit <- confusionMatrix(logit_pred, test$churned, positive = "Churned")
print(cm_logit)  

# ============================================================
# Decision tree (comparison model)
# ============================================================
tree_model <- rpart(churned ~ recency + frequency + monetary + tenure,
                    data = train, method = "class",
                    control = rpart.control(cp = 0.01))

rpart.plot(tree_model, type = 4, extra = 102,
           main = "Decision Tree – Customer Churn Prediction")
# ← SAVE THIS PLOT

tree_pred <- predict(tree_model, newdata = test, type = "class")
cm_tree   <- confusionMatrix(tree_pred, test$churned, positive = "Churned")
print(cm_tree)

# ============================================================
# ROC curve and AUC
# ============================================================
roc_logit <- roc(
  response  = ifelse(test$churned == "Churned", 1, 0),
  predictor = logit_probs
)

cat("Logistic Regression AUC:", round(auc(roc_logit), 3), "\n") 

plot(roc_logit, col = "#3266ad", lwd = 2,
     main = paste("ROC Curve – Logistic Regression | AUC =", round(auc(roc_logit), 3)))
abline(a = 0, b = 1, lty = 2, col = "gray")


# ============================================================
# Variable importance (odds ratios)
# ============================================================
exp(coef(logit_model))  

