# Paquetes ----
library(tidytuesdayR)
library(skimr)
library(themis)
library(tidyverse)
library(tidymodels)

# Datos ----
tt_data <- tidytuesdayR::tt_load(2020, week =39)
tt_data$members |> 
  skimr::skim() 
## Climbers DF ----
climbers_df <- tt_data$members |> 
  select(member_id, peak_name, season, year, sex, age, citizenship,expedition_role, hired, solo, oxygen_used, success, died) |> 
filter((!is.na(sex) & !is.na(citizenship) & !is.na(peak_name) & !is.na(expedition_role)) == T) |> 
  mutate(across(where(~ is.character(.) | is.logical(.)), as.factor))

# Data Split ----
## Semilla ----
set.seed(2023)
## Split inicial ----
climbers_split <- initial_split(climbers_df, prop = 0.8, strata = died)
climbers_split
## Conjunto de entrenamiento ----
train_set <- training(climbers_split)
## Conjunto de prueba ----
test_set <- testing(climbers_split)
## CV ----
climbers_fold <- train_set |> 
  vfold_cv(v = 10, repeats = 1, strata = died)

# Receta ----
mod_recipe <- recipe(formula = died ~ ., data = train_set)

mod_recipe <- mod_recipe |>
  update_role(member_id, new_role = "id") |> 
  step_impute_median(age) |> 
  step_normalize(all_numeric_predictors()) |>
  step_other(peak_name, citizenship, expedition_role, threshold = 0.05) |> 
  step_dummy(all_predictors(), -all_numeric(), one_hot = F) |> 
  step_upsample(died, over_ratio = 0.2, seed = 2023, skip = T) 
## Preparación ----
mod_recipe_prepped <- prep(mod_recipe, retain = T)
## Bake ----
bake(mod_recipe_prepped, new_data = NULL)

# Modelos ----
## Regresión logística glm ----
log_cls <- logistic_reg() |> 
  set_engine('glm') |> 
  set_mode("classification")
log_cls

## Regresión logística glmnet ----
reg_log_cls <- logistic_reg() |>
  set_args(penalty = tune(), mixture = tune()) |>  set_mode("classification") |> 
  set_engine("glmnet", family ="binomial")
reg_log_cls

## Workflow ----

cls_wf <- workflow() |> 
  add_recipe(mod_recipe) |> 
  add_model(reg_log_cls)
cls_wf

## Parámetros ----
param_grid <- grid_regular(
  penalty(), mixture(),
  levels = c(10,10)
)

param_grid |> glimpse()

## Ajuste ----
start <- Sys.time()
cls_wf_fit <- tune_grid(
  cls_wf, climbers_fold,
  grid = param_grid,
  metrics = metric_set(roc_auc, accuracy, sens, spec),
  control = control_grid(save_pred = T, verbose = T)
)
Sys.time() - start

## Desempeño ----
cls_wf_fit |> collect_metrics(summarize = T)
cls_wf_fit |>  show_best(metric = "roc_auc", n = 3)
cls_wf_fit |>  select_best(metric = "roc_auc")

cls_wf_fit %>% 
  collect_predictions(
    summarize = F, parameters = select_best(cls_wf_fit, metric = "roc_auc")
  )
## Gráfico de desempeño ----
autoplot(cls_wf_fit)

## Finalización del flujo ----
cls_wf_final <- cls_wf %>% 
  finalize_workflow(select_best(cls_wf_fit, metric = "roc_auc"))
cls_wf_final

## Predicción ----
cls_wf_last_fit <- cls_wf_final %>% 
  last_fit(split = climbers_split, metrics = metric_set(roc_auc, accuracy, sens, spec))
cls_wf_last_fit

# Entrega de resultados ----
library(broom)
library(tune)
reg_log_cls_fit <- cls_wf_last_fit |> extract_fit_parsnip()

## Componentes del modelo ----
tidy(reg_log_cls_fit) |> glimpse()
## Diagnóstico del modelo
glance(reg_log_cls_fit) |>  glimpse()
## Matriz de confusión ----
collect_predictions(cls_wf_last_fit) |> 
  conf_mat(died, estimate = .pred_class)
## Sensibilidad ----
collect_predictions(cls_wf_last_fit) |> sens(died, estimate = .pred_class)
## Especificidad ----
collect_predictions(cls_wf_last_fit) |>  
  spec(died, estimate = .pred_class)
## Precisión ----
collect_predictions(cls_wf_last_fit) |> 
  accuracy(died, estimate = .pred_class)
## Todas ----
metrics <- metric_set(accuracy, sens, spec)
collect_predictions(cls_wf_last_fit) |>  
  metrics(died, estimate = .pred_class)
## Curvas ROC ----
collect_predictions(cls_wf_fit) |> 
  group_by(id) |> 
  roc_curve(
    died, .pred_TRUE,
    event_level = "second"
  ) |> 
  autoplot()
