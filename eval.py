from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, RocCurveDisplay

# -------------------------------------------------
# PHASE: Final Prediction + Evaluation per Fold
# -------------------------------------------------

# Collect true and predicted values for ROC & CM
y_true_all = []
y_pred_all = []
y_score_all = []

for train_idx, test_idx in kf.split(X, Y):
    model = create_model(neurons=512, act_f=act_f, hiddenlayers=5, ki=ki, l1_value=l1_value, l2_value=l2_value,
                         dropout_rate=dropout_rate, learning_rate=learning_rate)
    model.fit(x=X[train_idx], y=Y[train_idx], epochs=20, batch_size=128, verbose=0)
    y_pred = model.predict(X[test_idx])
    y_class = (y_pred > 0.5).astype(int).flatten()

    y_true_all.extend(Y[test_idx])
    y_pred_all.extend(y_class)
    y_score_all.extend(y_pred.flatten())

# -------------------------------------------------
# 📊 CONFUSION MATRIX
# -------------------------------------------------
cm = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("lstm_results/cm_{}.png".format(fusion))

# -------------------------------------------------
# 📝 CLASSIFICATION REPORT
# -------------------------------------------------
report = classification_report(y_true_all, y_pred_all, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("lstm_results/report_{}.csv".format(fusion), index=True)

# -------------------------------------------------
# 📈 ROC CURVE & AUC
# -------------------------------------------------
fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.savefig("lstm_results/roc_{}.png".format(fusion))

# -------------------------------------------------
# 🧾 Final Print Summary
# -------------------------------------------------
print("\n[Summary: {}]".format(fusion))
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(y_true_all, y_pred_all))
print("ROC AUC Score: {:.4f}".format(roc_auc))
