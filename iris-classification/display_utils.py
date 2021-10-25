import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def metric_accuracy(y_true, y_pred, pos_label=True):
    return metrics.accuracy_score(y_true, y_pred)

def metric_f1(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average="weighted", pos_label=pos_label)

def metric_f1_micro(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average="micro", pos_label=pos_label)

def metric_f1_macro(y_true, y_pred, pos_label=True):
    return metrics.f1_score(y_true, y_pred, average="macro", pos_label=pos_label)

def metric_precision(y_true, y_pred, pos_label=True):
    return metrics.precision_score(y_true, y_pred, average="binary", pos_label=pos_label)

def metric_recall(y_true, y_pred, pos_label=True):
    return metrics.recall_score(y_true, y_pred, average="binary", pos_label=pos_label)

def metric_auc(y_true, y_pred, pos_label=True):
    try:
        fpr, tpr, _ = metrics.roc_curve(
            y_true,
            y_pred,
            pos_label=pos_label
        )
        return metrics.auc(fpr, tpr)
    except:
        return None

def metric_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def evaluate_classification(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        "ACCURACY": metric_accuracy(y_true, y_pred),
        "F1-SCORE": metric_f1(y_true, y_pred),
        "F1-MICRO": metric_f1_micro(y_true, y_pred) ,
        "F1-MACRO": metric_f1_macro(y_true, y_pred),
        "PRECISION": metric_precision(y_true, y_pred),
        "RECALL": metric_recall(y_true, y_pred),
        "AUC": metric_auc(y_true, y_score),
        "KAPPA": metric_kappa(y_true, y_pred),
        "TRUE_POSITIVE": tp,
        "TRUE_NEGATIVE": tn,
        "FALSE_POSITIVE": fp,
        "FALSE_NEGATIVE": fn
    }

    return pd.DataFrame({k: [v] for k, v in metrics_dict.items() if v is not None}, index=["Metrics"])


def binary_classification_evaluation_plot(model, X, y_true):
    """Function to display four visualizations to evaluate model's performance: ROC-AUC, CAP, 
    Precision-Recall and KS.
		
		Args: 
			model (sklearn.Pipeline): complete preprocessing amd model building pipeline
            X (pandas.DataFrame): dataset with all predictors
            y_true (pandas.Series): true labels
		
		Returns: 
			None
	
	""" 
    try:
        y_proba = model.predict_proba(X)[:,1]
    except:
        raise ValueError("Model does not return probabilities!")

    if len(np.unique(y_true))!=2:
        raise ValueError("Multiclass Problem!")

    fig, ax = plt.subplots(2,2,figsize=(12,8))
    plt.suptitle("Binary Classification Evaluation")
    __plot_roc(y_true, y_proba, ax[0][0])
    __plot_pr(y_true, y_proba, ax[0][1])
    __plot_cap(y_true, y_proba, ax[1][0])
    __plot_ks(y_true, y_proba, ax[1][1])
    plt.tight_layout()
    return fig

def __plot_cap(y_test, y_proba, ax):
    cap_df = pd.DataFrame(data=y_test, index=y_test.index)
    cap_df["Probability"] = y_proba

    total = cap_df.iloc[:, 0].sum()
    perfect_model = (cap_df.iloc[:, 0].sort_values(ascending=False).cumsum()/total).values
    current_model = (cap_df.sort_values(by="Probability", ascending=False).iloc[:, 0].cumsum()/total).values

    max_area = 0
    covered_area = 0
    h = 1/len(perfect_model)
    random = np.linspace(0, 1, len(perfect_model))
    for i, (am, ap) in enumerate(zip(current_model, perfect_model)):
        try:
            max_area += (ap-random[i]+perfect_model[i+1]-random[i+1])*h/2
            covered_area += (am-random[i]+current_model[i+1]-random[i+1])*h/2
        except:
            continue
    accuracy_ratio = covered_area/max_area

    ax.plot(np.linspace(0, 1, len(current_model)), current_model, 
                        color="green", label=f"AR = {accuracy_ratio:.3f}")
    ax.plot(np.linspace(0, 1, len(perfect_model)), perfect_model, color="red", label="Perfect Model")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("Individuals", fontsize=12)
    ax.set_ylabel("Target Individuals", fontsize=12)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.01))
    ax.legend(loc=4, fontsize=10)
    ax.set_title("CAP Analysis", fontsize=13)

def __plot_roc(y_test, y_proba, ax):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)

    ax.plot(fpr, tpr, color="red", label=f"(AUC = {metric_auc(y_test, y_proba):.3f})")
    ax.plot([0,1], [0,1], color="navy")
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("ROC Analysis", fontsize=13)

def __plot_pr(y_test, y_proba, ax):
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_proba)

    ax.plot(recall, precision, color="red", label=f"PR")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1.001))
    ax.legend(loc=4)
    ax.set_title("Precision-Recall Analysis", fontsize=13)

def __plot_ks(y_test, y_proba, ax):
    prediction_labels = pd.DataFrame(y_test.values, columns=["True Label"])
    prediction_labels["Probabilities"] = y_proba
    prediction_labels["Thresholds"] = prediction_labels["Probabilities"].apply(lambda x: np.round(x, 2))
    df = prediction_labels.groupby("Thresholds").agg(["count", "sum"])[["True Label"]]
    ks_df = pd.DataFrame(df["True Label"]["sum"]).rename(columns={"sum":"Negative"})
    ks_df["Positive"] = df["True Label"]["count"]-df["True Label"]["sum"]
    ks_df["Negative"] = ks_df["Negative"].cumsum()/ks_df["Negative"].sum()
    ks_df["Positive"] = ks_df["Positive"].cumsum()/ks_df["Positive"].sum()
    ks_df["KS"] = ks_df["Positive"]-ks_df["Negative"]
    ks_df.loc[0.0, :] = [0.0, 0.0, 0.0]
    ks_df = ks_df.sort_index()
    max_ks_thresh = ks_df.KS.idxmax()

    ks_df.drop("KS", axis=1).plot(color=["red", "navy"], ax=ax)
    ax.set_xlabel("Thresholds", fontsize=12)
    ax.set_xlabel("Target Individuals", fontsize=12)
    ax.set_title("KS Analysis", fontsize=13)
    ax.plot([max_ks_thresh, max_ks_thresh], 
            [ks_df.loc[max_ks_thresh,"Negative"], ks_df.loc[max_ks_thresh,"Positive"]],
            color="green", label="Max KS")
    ax.text(max_ks_thresh-0.16, 0.5, f"KS={ks_df.loc[max_ks_thresh,'KS']:.3f}", fontsize=12, color="green")
    ax.legend()