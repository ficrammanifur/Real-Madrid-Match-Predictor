import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib

def detailed_evaluation(model_path='real_madrid_model.pkl', data_path='real_madrid_matches.csv'):
    """
    Perform detailed model evaluation with visualizations
    """
    # Load model and data
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_engineer = model_data['feature_engineer']
    target_encoder = model_data['target_encoder']
    
    df = pd.read_csv(data_path)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Prepare data
    X, y, _ = feature_engineer.prepare_features(df)
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Draw', 'Loss', 'Win'],
                yticklabels=['Draw', 'Loss', 'Win'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC Curves for each class
    plt.subplot(2, 3, 2)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    colors = ['blue', 'red', 'green']
    class_names = ['Draw', 'Loss', 'Win']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    # 3. Prediction Confidence Distribution
    plt.subplot(2, 3, 3)
    max_proba = np.max(y_pred_proba, axis=1)
    plt.hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Maximum Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    
    # 4. Performance by Competition
    plt.subplot(2, 3, 4)
    df_test = df.iloc[split_idx:].copy()
    df_test['predicted'] = target_encoder.inverse_transform(y_pred)
    df_test['actual'] = target_encoder.inverse_transform(y_test)
    df_test['correct'] = df_test['predicted'] == df_test['actual']
    
    comp_accuracy = df_test.groupby('competition')['correct'].mean()
    comp_accuracy.plot(kind='bar')
    plt.title('Accuracy by Competition')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # 5. Performance by Venue
    plt.subplot(2, 3, 5)
    venue_accuracy = df_test.groupby('venue')['correct'].mean()
    venue_accuracy.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Accuracy by Venue')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0)
    
    # 6. Prediction Probability vs Actual Outcome
    plt.subplot(2, 3, 6)
    win_proba = y_pred_proba[:, 2]  # Win probabilities
    actual_wins = (y_test == 2).astype(int)
    
    # Bin predictions and calculate actual win rate
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_rates = []
    
    for i in range(len(bins)-1):
        mask = (win_proba >= bins[i]) & (win_proba < bins[i+1])
        if mask.sum() > 0:
            actual_rates.append(actual_wins[mask].mean())
        else:
            actual_rates.append(0)
    
    plt.plot(bin_centers, actual_rates, 'o-', label='Actual Win Rate')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Calibration Plot - Win Predictions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("=== Detailed Evaluation Results ===\n")
    
    print("Overall Accuracy by Class:")
    for i, class_name in enumerate(['Draw', 'Loss', 'Win']):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_accuracy = (y_pred[class_mask] == i).mean()
            print(f"{class_name}: {class_accuracy:.3f} ({class_mask.sum()} samples)")
    
    print(f"\nCompetition Performance:")
    for comp in df_test['competition'].unique():
        comp_mask = df_test['competition'] == comp
        comp_acc = df_test[comp_mask]['correct'].mean()
        print(f"{comp}: {comp_acc:.3f} ({comp_mask.sum()} matches)")
    
    print(f"\nVenue Performance:")
    for venue in df_test['venue'].unique():
        venue_mask = df_test['venue'] == venue
        venue_acc = df_test[venue_mask]['correct'].mean()
        print(f"{venue}: {venue_acc:.3f} ({venue_mask.sum()} matches)")
    
    return df_test

if __name__ == "__main__":
    detailed_evaluation()
