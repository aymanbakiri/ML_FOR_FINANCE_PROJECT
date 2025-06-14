from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix, 
    accuracy_score
)

import numpy as np
import torch
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



def report(y_test, y_pred, X_test, best_model, feature_columns):
    # Prediction & Classification Report
    y_pred = best_model.predict(X_test)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    # ROC AUC + Precision‐Recall AUC
    y_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    print(f"Test ROC AUC:  {roc_auc:.4f}")
    print(f"Test PR AUC:   {pr_auc:.4f}")

    # Display a confusion matrix if you like
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (low values = better balance):\n", cm)


    importances = best_model.named_steps['clf'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10))

    # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # print("\nRegression Metrics on Test Set:")
    # print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    # print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
    # print(f"R²:   {r2_score(y_test, y_pred):.4f}")


def train(epochs, model, scheduler, 
          clip_grad, optimizer, patience, 
          criterion, dl_val, dl_train, device, early_stop):
    for epoch in range(1, 101):
        model.train()
        losses = []
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                preds.extend(probs)
                trues.extend(yb.numpy())
        auc_val = roc_auc_score(trues, preds)
        scheduler.step(auc_val)
        print(f"Epoch {epoch:02d} | Loss {np.mean(losses):.4f} | Val AUC {auc_val:.4f}")
        if auc_val > best_auc:
            best_auc, trials = auc_val, 0
            torch.save(model.state_dict(), 'best_lstm.pth')
        else:
            trials += 1
            if (trials >= patience) and early_stop:
                print("Early stopping.")
                break


def evaluate(model, dl_test, device, inception=False):
    if inception:
        model.load_state_dict(torch.load('best_inception.pth'))
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            test_preds.extend(torch.sigmoid(model(xb)).cpu().numpy())
            test_trues.extend(yb.numpy())

    y_hat = (np.array(test_preds) > 0.5).astype(int)

    print("\nClassification Report on Test Set:")
    print(classification_report(test_trues, y_hat, digits=4))
    print("Confusion Matrix (low values = better balance):")
    print(confusion_matrix(test_trues, y_hat))

    roc = roc_auc_score(test_trues, test_preds)
    precision, recall, _ = precision_recall_curve(test_trues, test_preds)
    pr_auc = auc(recall, precision)
    acc = accuracy_score(test_trues, y_hat)

    print(f"\nTest ROC AUC:  {roc:.4f}")
    print(f"Test PR AUC:   {pr_auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")


def build_sequences(df, feature_columns, label_column, group_key, window):
    """
    Slides a fixed‐length window over each group in df and returns
    X (num_samples × window × num_features) and y (num_samples,).
    """
    Xs, ys = [], []
    for _, grp in df.groupby(group_key):
        grp = grp.sort_index()
        arr = grp[feature_columns].values
        labels = grp[label_column].values
        for i in range(window, len(arr)):
            Xs.append(arr[i-window:i])
            ys.append(labels[i])
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys).astype(np.float32)
    return X, y

def split_data(X, y, train_frac=0.8, val_frac=0.2):
    """
    Chronological split into train/val/test.
    train_frac is fraction of ALL data to use for train+val;
    val_frac is fraction *of the train+val* to hold out for val.
    """
    n = len(y)
    cut = int(n * train_frac)
    X_train_all, X_test = X[:cut], X[cut:]
    y_train_all, y_test = y[:cut], y[cut:]
    val_cut = int(len(X_train_all) * (1 - val_frac))
    X_train, X_val = X_train_all[:val_cut], X_train_all[val_cut:]
    y_train, y_val = y_train_all[:val_cut], y_train_all[val_cut:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def make_dataloaders(splits, batch_size, shuffle_train=True):
    """
    Given ((X_train,y_train), (X_val,y_val), (X_test,y_test)) and batch_size,
    return train/val/test DataLoaders.
    """
    to_tensor = lambda a: torch.from_numpy(a)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    dl_test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return dl_train, dl_val, dl_test

def make_loss(y_train):
    """
    Compute a pos_weight for BCEWithLogitsLoss from your binary labels.
    """
    pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum())
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    