import os, io, base64, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from flask import Flask, render_template, request, jsonify

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "dataset.csv")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    PKL = pickle.load(f)
model    = PKL["model"]
FEATURES = PKL["features"]

print("📂 Загружаем датасет...")
df_full   = pd.read_csv(DATA_PATH)
df_full["Extracurricular Activities"] = df_full["Extracurricular Activities"].map({"Yes": 1, "No": 0})
df_sample = df_full.sample(1000, random_state=7).reset_index(drop=True)
df_table  = df_full.head(200).copy()
df_table["Extracurricular Activities"] = df_table["Extracurricular Activities"].map({1: "Yes", 0: "No"})
print(f"   Готово. {len(df_full):,} строк.")


def make_chart(highlight=None):
    num_features = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]
    fig = plt.figure(figsize=(14, 6), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    labels = {
        "Hours Studied": "Часы учёбы",
        "Previous Scores": "Предыдущие оценки",
        "Sleep Hours": "Часов сна",
        "Sample Question Papers Practiced": "Пробных тестов",
    }

    from sklearn.linear_model import LinearRegression as LR

    for idx, feat in enumerate(num_features):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("#0f0f1a")
        for spine in ax.spines.values():
            spine.set_color("#252540")
        ax.tick_params(colors="#64748b", labelsize=7)

        x_vals = df_sample[feat].values
        y_vals = df_sample["Performance Index"].values

        ax.scatter(x_vals, y_vals, color="#818cf8", alpha=0.2, s=8, zorder=2)

        lr = LR().fit(x_vals.reshape(-1, 1), y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, lr.predict(x_line.reshape(-1,1)),
                color="#4ade80", linewidth=2, zorder=3)

        if highlight:
            hx = highlight["inputs"].get(feat)
            hy = highlight["score"]
            if hx is not None:
                ax.scatter([hx], [hy], color="#f472b6", s=100, zorder=5,
                           edgecolors="white", linewidths=1.2)

        ax.set_xlabel(labels[feat], color="#64748b", fontsize=7.5)
        ax.set_ylabel("Performance Index" if idx == 0 else "", color="#64748b", fontsize=7.5)
        ax.grid(color="#252540", linestyle="--", linewidth=0.5, alpha=0.8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.route("/")
def index():
    chart_b64 = make_chart()
    return render_template(
        "index.html",
        chart_b64  = chart_b64,
        r2         = round(PKL["r2"], 4),
        rmse       = round(PKL["rmse"], 2),
        coef       = PKL["coef"],
        intercept  = round(PKL["intercept"], 4),
        features   = FEATURES,
        total_rows = f"{len(df_full):,}",
        dataset    = df_table.to_dict(orient="records"),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        inputs = {
            "Hours Studied":                    float(data["hours"]),
            "Previous Scores":                  float(data["prev_scores"]),
            "Extracurricular Activities":        int(data["extra"]),
            "Sleep Hours":                      float(data["sleep"]),
            "Sample Question Papers Practiced":  float(data["papers"]),
        }
        X_in  = pd.DataFrame([inputs])[FEATURES]
        score = float(model.predict(X_in)[0])
        score = max(0, min(100, score))

        if   score >= 90: grade = "A"
        elif score >= 75: grade = "B"
        elif score >= 60: grade = "C"
        elif score >= 40: grade = "D"
        else:             grade = "F"

        chart = make_chart(highlight={"inputs": inputs, "score": score})
        return jsonify({"score": round(score, 1), "grade": grade, "chart_b64": chart})
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
