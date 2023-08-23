
####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
library (usethis)
create_github_token()
library (gitcreds)
gitcreds::gitcreds_set()
use_github()




library(tidyverse)
###para usar phyton inserto
library(reticulate)
use_Phyton ()

#### modelos
library(tidymodels)
##paquetes
#1)
parsnip ##define los modelos
parsnip_addin()##ver cuales son los modelos

#2)
recipes ###receta de como hecerlo

### ejemplo en ves de datos iris 
library(palmerpenguins)

penguins

penguins%>%
  filter(!is.na(sex))%>%
  ggplot(aes(flipper_length_mm, bill_length_mm,color=sex, size=body_mass_g))+
  geom_point(alpha=0.5)+
  facet_wrap(~species)

penguins.df<-penguins%>%
  filter(!is.na(sex)) |> 
  select(-island)

library(tidymodels)
set.seed (123)
penguins_split<- initial_split(penguins.df, strata=sex)
penguins_train<- training(penguins_split)
penguins_test<-testing(penguins_split)
penguins_split

###remuestreo de datos es opcional
set.seed(123)
penguins_boot<- bootstraps(penguins_train)
penguins_boot


###definiendo modelos, se hara regresion logistica
glm_spec<-logistic_reg() |> 
  set_engine('glm')
glm_spec


##random forest
rf_spec<-rand_forest() |> 
  set_mode('classification') |> 
  set_engine('ranger')
rf_spec

###flujo de trabajo
penguins_wf<-workflow() |> 
  add_formula(sex~.)
penguins_wf

glm_rs<-penguins_wf |> 
  add_model(glm_spec) |> 
  fit_resamples(resamples=penguins_boot, control=control_resamples(save_pred = TRUE))
glm_rs


###evaluando modelos














