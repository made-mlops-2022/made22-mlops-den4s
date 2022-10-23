import os
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DISCRETE_FEATURES = ["sex",       # 2. `1` = male; `0` = female
                     "cp",        # 3. chest pain type (`0` = typ; `1` = atyp; `2` = non-anginal; `3` = asymptomatic)
                     "fbs",       # 6. (fasting blood sugar > 120 mg/dl) (`1` = true; `0` = false)
                     "restecg",   # 7. resting electrocard. results (`0`, `1` or `2`)
                     "exang",     # 9. exercise induced angina (`1` = yes; `0` = no)
                     "slope",     # 11. the slope of the peak exercise ST segment (`0` = up; `1` = flat; `2` = down)
                     "ca",        # 12. number of major vessels (`0-3`) colored by flourosopy
                     "thal"]      # 13. `0` = normal; `1` = fixed defect; `2` = reversable defect
CONTINUOUS_FEATURES = ["age",       # 1. age
                       "trestbps",  # 4. resting blood pressure in mmHg
                       "chol",      # 5. serum cholestoral in mg/dl
                       "thalach",   # 8. maximum heart rate achieved
                       "oldpeak"]   # 10. ST depression induced by exercise relative to rest
ANSWERS = "condition"  # `0` = no disease, `1` = disease
REP_DIR = "reports"  # directory for reports


def describe_continuous_columns(df: pd.DataFrame) -> NoReturn:
    desc = df[CONTINUOUS_FEATURES].describe()
    desc.to_csv(os.path.join(REP_DIR, "continuous_features_describe.csv"))


def describe_discrete_columns(df: pd.DataFrame) -> NoReturn:
    desc = df[DISCRETE_FEATURES].astype("category").describe()
    desc.to_csv(os.path.join(REP_DIR, "discrete_features_describe.csv"))


def pie_plot(df: pd.DataFrame) -> NoReturn:
    # pie-plot params
    font_size = 12
    ang_pie = 180
    rad_pie = 1
    no_disease_color = "lightgreen"
    disease_color = "lightcoral"
    no_disease_color_dark = "mediumseagreen"
    disease_color_dark = "indianred"

    disease_by_sex_stat = df.groupby([ANSWERS, "sex"]).size()
    plt.pie(disease_by_sex_stat,
            labels=["$female$", "$male$", "$female$", "$male$"],
            colors=[no_disease_color, no_disease_color_dark, disease_color, disease_color_dark],
            startangle=ang_pie, explode = (0, 0.1, 0, 0.1),
            autopct="%.2f%%", radius=rad_pie, textprops={"fontsize": font_size})
    plt.savefig(os.path.join(REP_DIR, "pie_disease_sex.png"))


def num_target_pairplot(df: pd.DataFrame) -> NoReturn:
    no_disease_color_dark = "mediumseagreen"
    disease_color_dark = "indianred"
    sns.pairplot(df, vars=CONTINUOUS_FEATURES, hue=ANSWERS,
                 palette={0: no_disease_color_dark, 1: disease_color_dark})
    plt.grid()
    plt.savefig(os.path.join(REP_DIR, "pair_plot_continuous_features.png"))


def main():
    plt.style.use("seaborn")
    os.makedirs(REP_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join("data", "raw", "heart_cleveland_upload.csv"))
    describe_continuous_columns(df)
    describe_discrete_columns(df)
    pie_plot(df)
    num_target_pairplot(df)


if __name__ == "__main__":
    main()
