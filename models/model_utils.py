
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from data_processing.utils import DataMissingError
from models.cyclic_model import modele_cyclique

def validation_croisee(model, X, y, cv=5, verbose=True):
    """
    Performs cross-validation for a given model and data.
    
    Args:
        model: The model to evaluate (sklearn compatible).
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target variable.
        cv (int or cross-validation generator): Cross-validation splitting strategy.
        verbose (bool): Whether to print results.
        
    Returns:
        dict: Dictionary containing mean and std of RMSE and Resolution (if applicable).
    """
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, mean_squared_error
    import numpy as np

    # Custom scorer for Resolution: std(100 * (pred - true) / true)
    def resolution_scorer(y_true, y_pred):
        return np.std(100 * (y_pred - y_true) / y_true)

    scoring = {
        'mse': 'neg_mean_squared_error',
        'resolution': make_scorer(resolution_scorer)
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    rmse_test = np.sqrt(-scores['test_mse'])
    rmse_train = np.sqrt(-scores['train_mse'])
    res_test = scores['test_resolution']
    res_train = scores['train_resolution']

    results = {
        'rmse_train_mean': np.mean(rmse_train),
        'rmse_train_std': np.std(rmse_train),
        'rmse_test_mean': np.mean(rmse_test),
        'rmse_test_std': np.std(rmse_test),
        'res_train_mean': np.mean(res_train),
        'res_train_std': np.std(res_train),
        'res_test_mean': np.mean(res_test),
        'res_test_std': np.std(res_test)
    }

    if verbose:
        print(f"Cross-Validation Results ({cv} folds):")
        print(f"RMSE Train: {results['rmse_train_mean']:.2f} ± {results['rmse_train_std']:.2f}")
        print(f"RMSE Test : {results['rmse_test_mean']:.2f} ± {results['rmse_test_std']:.2f}")
        print(f"Res Train : {results['res_train_mean']:.2f}% ± {results['res_train_std']:.2f}%")
        print(f"Res Test  : {results['res_test_mean']:.2f}% ± {results['res_test_std']:.2f}%")
    
    return results

def stepwise_bic(X, y):
    initial_features = []
    best_features = initial_features.copy()
    current_score = float('inf')
    best_new_score = current_score

    remaining = list(X.columns)
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            features_to_test = best_features + [candidate]
            X_model = sm.add_constant(X[features_to_test])
            model = sm.OLS(y, X_model).fit()
            bic = model.bic
            scores_with_candidates.append((bic, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score < current_score:
            remaining.remove(best_candidate)
            best_features.append(best_candidate)
            current_score = best_new_score
        else:
            break

    return best_features

def stepwise_aic(X, y):
    initial_features = []
    best_features = initial_features.copy()
    current_score = float('inf')
    best_new_score = current_score

    remaining = list(X.columns)
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            features_to_test = best_features + [candidate]
            X_model = sm.add_constant(X[features_to_test])
            model = sm.OLS(y, X_model).fit()
            aic = model.aic
            scores_with_candidates.append((aic, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score < current_score:
            remaining.remove(best_candidate)
            best_features.append(best_candidate)
            current_score = best_new_score
        else:
            break

    return best_features

def print_erreurs(liste_decoupe, critere_decoupe, target, selected_variables): 
    ncols = 3
    nrows = len(liste_decoupe)
    figsize = (15, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.reshape(nrows, ncols) 

    for i, decoupe in enumerate(liste_decoupe):
        decoupe = decoupe.copy()  # Pour éviter de modifier l'original
        model = LinearRegression()
        X = decoupe[selected_variables]
        y = decoupe[target]
        model.fit(X, y)
        
        #on calcul l'efficacité du modèle sur ces portion d'energie
        cv = KFold(n_splits=5, shuffle=True)

        y_pred_cv = pd.Series(cross_val_predict(model, X, y, cv=cv), index=X.index)

        resolution_cv = np.std(100 * (y_pred_cv - y) / y)
        biais = np.mean(100 * (y_pred_cv - y) / y)
        
        # Titre
        val_min = decoupe[critere_decoupe].min()
        val_max = decoupe[critere_decoupe].max()
        titre_decoupe = f"{critere_decoupe} ∈ [{val_min:.2f}, {val_max:.2f}]"
        
        #Pred vs TRUE
        ax_pred = axes[i, 0]

        # 2D histogramme avec échelle log pour les counts
        hb = ax_pred.hist2d(
            y, 
            y_pred_cv, 
            bins=10,          # ajustable
            cmap='Blues',      # colormap
            norm=colors.LogNorm()     # échelle log
        )

        # Ajouter la droite y = x
        ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')

        ax_pred.set_title(f'Prédictions vs Réelles pour {target} ({titre_decoupe})')
        ax_pred.set_xlabel('Valeurs réelles')
        ax_pred.set_ylabel('Valeurs prédites')
        ax_pred.grid(True)
        ax_pred.text(0.05, 0.95, f"Résolution : {resolution_cv:.2f}%", transform=ax_pred.transAxes, fontsize=7, verticalalignment='top', color='red')
        ax_pred.text(0.05, 0.90, f"Nb élém : {len(decoupe)}", transform=ax_pred.transAxes, fontsize=7, verticalalignment='top', color='red')

        # Ajouter colorbar pour visualiser les counts
        cbar = plt.colorbar(hb[3], ax=ax_pred)
                
        # HISTOGRAMME erreur relative
        relative_error = np.std(100 * (y_pred_cv - y) / y)
        biais = np.mean(100 * (y_pred_cv - y) / y)
        ax_hist = axes[i, 1]
        sns.histplot(100 * (y_pred_cv - y) / y, bins=100, kde=True, color='blue', ax=ax_hist, stat='probability')
        ax_hist.set_title(f'Histogramme des erreurs relatives pour {target}')
        ax_hist.set_xlabel('Erreur relative (%)')
        ax_hist.set_ylabel('Probabilité')
        ax_hist.grid(True)
        ax_hist.axvline(x=biais, color='red', linestyle='-', label=f'Biais : {biais:.2f}%')
        ax_hist.legend()
        
        # Résolution et biais par bin d'énergie, abscisse = centre des bins (valeurs de 0 à 1000 tous les 100)
        ax_res = axes[i, 2]
        bins = np.arange(0, 1001, 100)
        decoupe['energy_bin'] = pd.cut(decoupe['energy'], bins, include_lowest=True)
        # Res/biais par bin
        mean_res = decoupe.groupby('energy_bin', observed=False).apply(
            lambda x: np.std(100 * (y_pred_cv.loc[x.index] - y.loc[x.index]) / y.loc[x.index])
            if len(x) > 0 else np.nan
        ).dropna()

        mean_bias = decoupe.groupby('energy_bin', observed=False).apply(
            lambda x: np.mean(100 * (y_pred_cv.loc[x.index] - y.loc[x.index]) / y.loc[x.index])
            if len(x) > 0 else np.nan
        ).dropna()

        bin_centers = [interval.mid for interval in mean_res.index.categories if interval in mean_res.index]

        ax_bias = ax_res.twinx()
        ax_res.plot(bin_centers, mean_res.values, marker='o', color='blue', linestyle='-', alpha=0.7, label='Résolution')
        ax_bias.plot(bin_centers, mean_bias.values, marker='x', color='orange', linestyle='--', alpha=0.7, label='Biais')

        ax_res.set_title("Résolution et biais en fonction de l'énergie")
        ax_res.set_xlabel("Énergie")
        ax_res.set_ylabel("Résolution (%)", color='blue')
        ax_bias.set_ylabel("Biais (%)", color='orange')
        #axe couleur
        ax_bias.spines['right'].set_color('orange')
        ax_bias.yaxis.label.set_color('orange')
        ax_bias.tick_params(axis='y', colors='orange')

        ax_res.spines['left'].set_color('blue')
        ax_res.yaxis.label.set_color('blue')
        ax_res.tick_params(axis='y', colors='blue')

        ax_res.grid(True)
        ax_res.axhline(y=resolution_cv, color='red', linestyle=':', label=f'Résolution globale : {resolution_cv:.2f}%')
        ax_res.set_xticks(bin_centers)

        lines, labels = ax_res.get_legend_handles_labels()
        lines2, labels2 = ax_bias.get_legend_handles_labels()
        ax_res.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout()
    plt.show()

def print_relative_error(df, target, x_variables):
    
    model = LinearRegression()
    X = df[x_variables]
    y = df[target]
    model.fit(X, y)
    y_pred = model.predict(X)
    
    relative_error = np.std(100 * (y_pred - y) / y)
    
    #histogramme des erreurs relatives
    plt.figure(figsize=(8, 6))
    sns.histplot(100 * (y_pred - y) / y, bins=50, kde=True, color='blue')
    plt.title(f'Histogramme des erreurs relatives pour {target} vs {x_variables}')
    plt.xlabel('Erreur relative (%)')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.axvline(x=relative_error, color='red', linestyle='--', label=f'Erreur relative moyenne : {relative_error:.2f}%')
    plt.legend()
    plt.show()

def validation_croisee_modele_cyclique(df, x_variables_1, x_variables_2, x_variables_towall, target="energy", n_splits=10, test_size=0.3, verbose=True):
    X = df[np.unique(x_variables_1 + x_variables_2 + x_variables_towall)]
    y = df[target]
    
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
    resolution_energy_scores = []
    resolution_towall_scores = []
    mse_towall_scores = []
    resolution_energy_1_scores = []
    bins_energy = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1001]  # Tranches d'énergie
    
    res_per_bin_mean_2 = [[] for _ in range(len(bins_energy)-1)]
    res_per_bin_mean_1 = [[] for _ in range(len(bins_energy)-1)]

    for train_idx, test_idx in cv.split(X):
        try :
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            # Entraînement du modèle cyclique sur les données d'entraînement
            model = modele_cyclique(df_train, x_variables_1, x_variables_2, x_variables_towall)

            y_test = df_test[target].values
            y_test_towall = df_test['towall'].values
            y_pred_energy_2, y_pred_towall, y_pred_energy_1 = model.predict(df_test)
            res_per_bin_2 = [[] for _ in range(len(bins_energy)-1)]
            res_per_bin_1 = [[] for _ in range(len(bins_energy)-1)]
            for i, energy in enumerate(y_pred_energy_2):
                # Trouver l'index de la tranche d'énergie correspondante
                for j in range(len(bins_energy) - 1):
                    if bins_energy[j] <= energy <= bins_energy[j + 1]:
                        res_per_bin_2[j].append(100 * (y_pred_energy_2[i] - y_test[i]) / y_test[i])
                        break
                # Calcul des résolutions
            for i, res in enumerate(res_per_bin_2):
                if len(res) > 0:
                    res_per_bin_mean_2[i].append(np.std(res))
                else:
                    res_per_bin_mean_2[i].append(np.nan)
            
            for i, energy in enumerate(y_pred_energy_1):
                # Trouver l'index de la tranche d'énergie correspondante
                for j in range(len(bins_energy) - 1):
                    if bins_energy[j] <= energy <= bins_energy[j + 1]:
                        res_per_bin_1[j].append(100 * (y_pred_energy_1[i] - y_test[i]) / y_test[i])
                        break
            for i, res in enumerate(res_per_bin_1):
                if len(res) > 0:
                    res_per_bin_mean_1[i].append(np.std(res))
                else:
                    res_per_bin_mean_1[i].append(np.nan)

            
            resolution_energy = np.std(100 * (y_pred_energy_2 - y_test) / y_test)
            resolution_towall = np.std(100 * (y_pred_towall - y_test_towall) / y_test_towall)
            mse_towall = mean_squared_error(y_test_towall, y_pred_towall)
            resolution_energy_1 = np.std(100 * (y_pred_energy_1 - y_test) / y_test)
            
            resolution_energy_scores.append(resolution_energy)
            resolution_towall_scores.append(resolution_towall)
            mse_towall_scores.append(np.sqrt(mse_towall))
            resolution_energy_1_scores.append(resolution_energy_1)
        except DataMissingError as e:
            print(f"Erreur lors de l'entraînement du modèle : {e.message}")
            continue


    for i in range(len(res_per_bin_mean_2)):
        res_per_bin_mean_2[i] = np.nanmean(res_per_bin_mean_2[i]) if len(res_per_bin_mean_2[i]) > 0 else np.nan
    for i in range(len(res_per_bin_mean_1)):
        res_per_bin_mean_1[i] = np.nanmean(res_per_bin_mean_1[i]) if len(res_per_bin_mean_1[i]) > 0 else np.nan
    if verbose:    
        print(f"Validation croisée ({n_splits} splits) :")
        print(f"Résolution energy 1: {np.mean(resolution_energy_1_scores):.2f} ± {np.std(resolution_energy_1_scores):.2f} %")
        print(f"Résolution energy 2: {np.mean(resolution_energy_scores):.2f} ± {np.std(resolution_energy_scores):.2f} %")
        print(f"Résolution towall: {np.mean(resolution_towall_scores):.2f} ± {np.std(resolution_towall_scores):.2f} %")
        print(f"MSE towall: {np.mean(mse_towall_scores):.2f} ± {np.std(mse_towall_scores):.2f}")
        
        # Affichage des résolutions par tranche d'énergie pour les deux prédictions (energy_1 et energy_2)
        fig, ax = plt.subplots(figsize=(12, 6))
        bin_centers = [(bins_energy[i] + bins_energy[i+1]) / 2 for i in range(len(bins_energy) - 1)]
        ax.plot(bin_centers, res_per_bin_mean_1, marker='o', linestyle='-', color='orange', label='Résolution energy_1')
        ax.plot(bin_centers, res_per_bin_mean_2, marker='o', linestyle='-', color='blue', label='Résolution energy_2')
        ax.set_xlabel('Énergie (MeV)')
        ax.set_ylabel('Résolution (%)')
        ax.set_title('Résolution par tranche d\'énergie (energy_1 vs energy_2)')
        ax.grid(True)
        ax.legend()
        plt.show()
        
        # scatter plot de l'énergie prédite vs réelle avec l'info de l'énergie
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(y_test, y_pred_energy_2, c=df_test['towall'], cmap='viridis', alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        cbar = fig.colorbar(scatter, ax=ax, label='towall (cm)')
        ax.set_xlabel('Valeurs réelles')
        ax.set_ylabel('Valeurs prédites')
        ax.set_title('Prédictions d\'énergie vs Réelles')
        ax.grid(True)
        ax.text(0.05, 0.95, f"Résolution : {np.mean(resolution_energy_scores):.2f}%", transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')
        ax.text(0.05, 0.90, f"Nb élém : {len(df_test)}", transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')
        plt.show()
    
    return {
        "resolution_energy_1": np.mean(resolution_energy_1_scores),
        "resolution_energy_2": np.mean(resolution_energy_scores),
        "resolution_towall": np.mean(resolution_towall_scores),
        "mse_towall": np.mean(mse_towall_scores),
        "res_per_bin_mean_1": res_per_bin_mean_1,
        "res_per_bin_mean_2": res_per_bin_mean_2
        }

def print_res_modeles(df, x_variables_1, x_variables_2, x_variables_towall, target="energy", n_splits=10, test_size=0.3):
    results_cyclique = validation_croisee_modele_cyclique(df, x_variables_1, x_variables_2, x_variables_towall, target, n_splits, test_size, verbose=False)
    results_reg_simple = validation_croisee(df, "energy", "charge_totale", [], [], verbose=False)
    results_reg_charge_n_hits = validation_croisee(df, "energy", ["charge_totale", "n_hits", "max_charge", "min_charge"], [], [], verbose=False)
    
    #Res per bin d'energy
    bins_energy = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1001]  # Tranches d'énergie
    bin_centers = [(bins_energy[i] + bins_energy[i+1]) / 2 for i in range(len(bins_energy) - 1)]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_centers, results_cyclique["res_per_bin_mean_2"], marker='o', linestyle='-', color='orange', label='Modèle combiné')
    ax.plot(bin_centers, results_reg_simple, marker='o', linestyle='-', color='blue', label='Reg simple (charge_totale)')
    ax.plot(bin_centers, results_reg_charge_n_hits, marker='o', linestyle='-', color='green', label='Reg multiple')
    ax.set_xlabel('Énergie (MeV)')
    ax.set_ylabel('Résolution (%)')
    ax.set_title('Comparaison entre modèles des résolutions par tranche d\'énergie')
    ax.grid(True)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def fit_exp_angle_dist_vertex(single_event, verbose ):
    """
    Renvoie le parametre lambda après le fit de single_event
    """
    # 1. Définir la fonction exponentielle décroissante
    def exp_decay(x, A, lamb):
        return A * np.exp(-x / lamb)

    # 2. Récupérer les données et nettoyer
    x = single_event["distance_vertex_to_hit"].values
    y = single_event["charge"].values

    # Masque pour éviter les NaN, inf et charges <= 0
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    # 3. Regrouper les données par bins de distance
    df_fit = pd.DataFrame({'x': x_clean, 'y': y_clean})
    df_fit['x_bin'] = pd.cut(df_fit['x'], bins=40)  # 40 bins adaptables

    # Moyenne dans chaque bin
    binned = df_fit.groupby('x_bin', observed=True).agg({'x': 'max', 'y': 'max'}).dropna()

    x_binned = binned['x'].values
    y_binned = binned['y'].values

    # 4. Fit exponentiel sur les points binned
    p0 = [np.max(y_binned), 300]  # estimation initiale : [A, λ]
    try:
        params, _ = curve_fit(exp_decay, x_binned, y_binned, p0=p0, bounds=([0, 1e-12], [np.inf, 5000]))
        A_fit, lambda_fit = params
        if verbose :
            print(f"✅ Fit réussi : A = {A_fit:.2f}, λ = {lambda_fit:.2f}")
    except RuntimeError:
        print("❌ Le fit a échoué choix de 1000 par défaut")
        lambda_fit = 1000

    if verbose :
        x_fit = np.linspace(np.min(x_binned), np.max(x_binned), 500)
        y_fit = exp_decay(x_fit, A_fit, lambda_fit)

        plt.figure(figsize=(10, 6))
        plt.scatter(x_clean, y_clean, s=2, alpha=0.1, label="Données brutes")
        plt.plot(x_binned, y_binned, 'o', color='black', label="Moyenne par bin")
        plt.plot(x_fit, y_fit, color='red', label=f"Fit: A·exp(-x/λ)\nλ = {lambda_fit:.1f}")
        plt.xlabel("Distance vertex → PMT")
        plt.ylabel("Charge")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return lambda_fit
