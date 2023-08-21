#### otra clase ----
library(tidytuesdayR)
library(skimr)
library(themis)
library(tidyverse)
library(tidymodels)

#### datos ----
tt_data<-tidytuesdayR::tt_load(2020, week =39)
tt_data$members |> 
  skimr::skim()

tt_data$members |> 
  skimr::skim()###resumen detallado

glimpse(tt_data$members)###resumen simple

climber_df<-tt_data$members |> 
  select(member_id, peak_name, season, year, sex,
         age, citizenship, expedition_role, hired, solo,
         oxygen_used,success,died) |> 
  filter ((!is.na(sex)& !is.na(citizenship)& !is.na(peak_name)&
            !is.na(expedition_role))==T ) |> 
  mutate(across(where(~is.character(.) |
                         is.logical(.)),as.factor))

### data split
climber_split<-initial_split(climber_df, prop=0.8,strata=died)

###semilla
set.seed(2023)


# Data Split ----
## Semilla ----
set.seed(2023)
## Split inicial ----
climber_split <- initial_split(climber_df, prop = 0.8, strata = died)
climber_split
## Conjunto de entrenamiento ----
train_set <- training(climber_split)
## Conjunto de prueba ----
test_set <- testing(climber_split)
###CV
climber_fold<-train_set |> 
  vfold_cv(v=10, repeats = 1, strata=died)

#recipe o receta----
mod_recipe<-recipe(formula=died~.,data=train_set)

mod_recipe<-mod_recipe |> 
  update_role(member_id, new_role = 'id') |> 
  step_impute_median(age) |>
  step_normalize(all_numeric_predictors()) |> 
  step_other(peak_name, citizenship, expedition_role, threshold = 0.05) |> 
  step_dummy(all_predictors(),-all_numeric(),one_hot = F) |> 
  step_upsample(died, over_ratio = 0.2, seed = 2023, skip = T)

####preparacion----
mod_recipe_prepped<-prep(mod_recipe, retain = T)

###bake hornear----
bake(mod_recipe_prepped, new_data = NULL)

####configuracion de modelos----
log_clas<-logistic_reg() |> 
  set_engine('glm') |> 
  set_mode('classification')

###regresion logistica
reg_log_cls<-logistic_reg() |> 
  set_args(penalty=tune(), mixture=tune()) |> 
  set_mode('classification')|>
  set_engine('glmnet', family='binomial')

####flujo de trabajo
cls_wf<-workflow() |> 
  add_recipe(mod_recipe) |> 
  add_model(reg_log_cls)
###parametros
param_grid<-grid_regular(penalty(), mixture(), levels=c(10,10))


###