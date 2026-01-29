
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
import random
from matplotlib import colors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from models.model_utils import validation_croisee

def distribution(df, liste_variables, bins=50):
    """
    Affiche la distribution des variables spécifiées dans le DataFrame sous forme de panel de graphiques.
    """
    ncols = 2
    nrows = len(liste_variables) // ncols + (len(liste_variables) % ncols > 0)
    figsize = (12, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, variable in enumerate(liste_variables):
        ax = axes[i]
        sns.histplot(df[variable], bins=bins, kde=True, ax=ax, color='blue')
        ax.set_title(f'Distribution de {variable}')
        ax.set_xlabel(variable)
        ax.set_ylabel('Fréquence')
        ax.grid(True)

    # Supprimer les axes inutilisés si len(liste_variables) < nrows * ncols
    for j in range(len(liste_variables), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    

def print_plot_reg(liste_découpe, critere_decoupe, cible, x_variable, df=None, reg=True, cmap=None): 
    """
    Affiche un panel de graphique, chaque graphique correspondant à un découpe particulière 
    est un plot de 'cible' en fonction de 'x_variable' coloré par le critère de découpe.
    L'argument 'reg' indique si on veux pour chaque graphique une régression linéaire.
    """
    # Fix: cmap default logic to avoid picking random every time if not specified
    if cmap is None:
        cmap = plt.get_cmap("cividis")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    ncols = 4
    nrows = len(liste_découpe) // ncols + (len(liste_découpe) % ncols > 0)
    figsize = (24, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Échelle de couleur globale si df fourni
    use_global_colorbar = df is not None
    if use_global_colorbar:
        vmin = df[critere_decoupe].min()
        vmax = df[critere_decoupe].max()

    scatter_list = []

    for i, (decoupe, ax) in enumerate(zip(liste_découpe, axes)):
        if use_global_colorbar:
            vmin_i, vmax_i = vmin, vmax
        else:
            vmin_i, vmax_i = decoupe[critere_decoupe].min(), decoupe[critere_decoupe].max()

        sc = ax.scatter(
            decoupe[cible],
            decoupe[x_variable],
            c=decoupe[critere_decoupe],
            cmap=cmap,
            vmin=vmin_i,
            vmax=vmax_i,
            s=10,
            alpha=0.6
        )
        scatter_list.append(sc)
        if reg :
            slope, intercept = np.polyfit(decoupe[cible], decoupe[x_variable], 1)
            x_vals = np.array(ax.get_xlim())
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, '--', color='red')

            ax.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}',
                        transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')
            # Corrected validation_croisee call
            # Predict x_variable (Y on plot) from cible (X on plot) to match polyfit
            X_val = decoupe[[cible]]
            y_val = decoupe[x_variable]
            model_val = LinearRegression()
            res_dict = validation_croisee(model_val, X_val, y_val, cv=5, verbose=False)
            
            # Key 'res_test_mean' contains the resolution metric
            res_val = res_dict.get('res_test_mean', None)
            
            if res_val is not None:
                ax.text(0.05, 0.90, f'res = {res_val:.2f}%',
                            transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')
        
        #Nombre d'éléments
        ax.text(0.05, 0.85, f'Nb élém = {len(decoupe)}',
                    transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')

        ax.set_title(f'Découpe {i+1} : {critere_decoupe} -> [{decoupe[critere_decoupe].min():.2f}, {decoupe[critere_decoupe].max():.2f}]')
        ax.set_xlabel(cible)
        ax.set_ylabel(x_variable)
        ax.grid(True)

        # Colorbar locale si pas de global
        if not use_global_colorbar:
            fig.colorbar(sc, ax=ax, orientation='vertical', label=critere_decoupe)

    # Supprimer les axes inutilisés si len(liste_découpe) < nrows * ncols
    for j in range(len(liste_découpe), len(axes)):
        fig.delaxes(axes[j])

    # Ajouter colorbar globale sur le côté droit
    if use_global_colorbar and scatter_list:
        # Ajuste les marges pour faire de la place à droite
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        fig.colorbar(scatter_list[0], cax=cbar_ax, label=critere_decoupe)

    plt.suptitle(f"{cible} vs {x_variable} avec couleur selon {critere_decoupe}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])  # Ajuste à l’espace laissé pour la colorbar
    plt.show()

def print_plot_reg_poly(liste_découpe, critere_decoupe, cible, x_variable, df=None, reg=True, cmap=None, degree=2): 
    """
    Affiche un panel de graphique, chaque graphique correspondant à un découpe particulière 
    est un plot de 'cible' en fonction de 'x_variable' coloré par le critère de découpe.
    L'argument 'reg' indique si on veux pour chaque graphique une régression polynomiale au 'degree' précisé.
    """
    if cmap is None:
        cmap = plt.get_cmap("cividis")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    ncols = 4
    nrows = len(liste_découpe) // ncols + (len(liste_découpe) % ncols > 0)
    figsize = (26, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    use_global_colorbar = df is not None
    if use_global_colorbar:
        vmin = df[critere_decoupe].min()
        vmax = df[critere_decoupe].max()

    scatter_list = []

    for i, (decoupe, ax) in enumerate(zip(liste_découpe, axes)):
        if use_global_colorbar:
            vmin_i, vmax_i = vmin, vmax
        else:
            vmin_i, vmax_i = decoupe[critere_decoupe].min(), decoupe[critere_decoupe].max()

        sc = ax.scatter(
            decoupe[x_variable],
            decoupe[cible],
            c=decoupe[critere_decoupe],
            cmap=cmap,
            vmin=vmin_i,
            vmax=vmax_i,
            s=10,
            alpha=0.6
        )
        scatter_list.append(sc)

        if reg:
            X = decoupe[[x_variable]].values
            y = decoupe[cible].values

            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            # Prédictions sur les données d'entraînement (pour le calcul du RMSE)
            y_pred = model.predict(X_poly)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Courbe du modèle
            x_vals = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            x_vals_poly = poly.transform(x_vals)
            y_vals = model.predict(x_vals_poly)
            ax.plot(x_vals, y_vals, '--', color='red')


            # Affichage du RMSE correct
            ax.text(0.70, 0.10, f'Root MSE = {rmse:.2f}', transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')

            # Nb d'éléments
            ax.text(0.70, 0.05, f'Nb élém = {len(decoupe)}', transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')

        ax.text(0.70, 0.05, f'Nb élém = {len(decoupe)}',
                transform=ax.transAxes, fontsize=7, verticalalignment='top', color='red')

        ax.set_title(f'Découpe {i+1} : {critere_decoupe} -> [{decoupe[critere_decoupe].min():.2f}, {decoupe[critere_decoupe].max():.2f}]')
        ax.set_xlabel(x_variable)
        ax.set_ylabel(cible)
        ax.grid(True)

        if not use_global_colorbar:
            fig.colorbar(sc, ax=ax, orientation='vertical', label=critere_decoupe)

    if use_global_colorbar and scatter_list:
        fig.colorbar(scatter_list[0], ax=axes.ravel().tolist(), label=critere_decoupe)

    plt.suptitle(f"{cible} vs {x_variable} avec fit polynomial (degré {degree})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def print_corr_reg_infos(liste_découpe_to_wall, target, x_variable, visuelize=True):
    
    resolution_overall, MSE_overall = [], []
    for i, decoupe in enumerate(liste_découpe_to_wall):
        corr_matrix = decoupe.corr()
        corr = corr_matrix.loc[target, x_variable]
        if not visuelize:
            print("-----------------------------------------------------\n-----------------------------------------------------")
            print(f"Découpe {i+1} :")
            print(f"Bounds : [{decoupe['towall'].min():.2f}, {decoupe['towall'].max():.2f}]")
            print(f"Corrélation entre {target} et {x_variable} : {corr:.3f}")
            print(f"Nombre d'éléments : {len(decoupe)}") 
        if len(decoupe) > 5: # Need enough samples for CV
             X_val = decoupe[[x_variable]]
             y_val = decoupe[target]
             model_val = LinearRegression()
             res_dict = validation_croisee(model_val, X_val, y_val, cv=5, verbose=False)
             if 'res_test_mean' in res_dict:
                 resolution_overall.append(res_dict['res_test_mean'])
                 # MSE is usually negative in sklearn scoring dict inside validation_croisee but returned as positive sqrt(mse) in results?
                 # validation_croisee returns rmse. function expects MSE list? or just list to append?
                 # original code was: validation_croisee(..., resolution_overall, MSE_overall, ...)
                 # Assuming we want to track metric.
                 # Let's verify what print_corr_reg_infos does with MSE_overall. It doesn't seem to use it in the snippet I see.
                 # But it plots resolution_overall.
             else:
                 resolution_overall.append(np.nan)
        else:
             resolution_overall.append(np.nan)
        
    if visuelize:
        # Visualisation des R² en fonction des bounds
        bounds = [f"[{decoupe['towall'].min():.2f}, {decoupe['towall'].max():.2f}]" for decoupe in liste_découpe_to_wall]
        # Création des barres et lignes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        bars = ax1.plot(bounds, resolution_overall, color='b',marker = 'x', label='Resolution Test')
        ax1.set_xlabel('Bounds')
        ax1.set_xticklabels(bounds, rotation=45, ha='right')
        ax1.set_ylabel('Resolution Test', color='b')
        ax1.tick_params(axis='y', labelcolor='b')


        # Ligne seuils
        ax1.axhline(y=25, color='g', linestyle='--', label='Seuil acceptable Resolution = 25')
        ax1.axhline(y=15, color='r', linestyle='--', label='Seuil significatif Resolution = 15')


        ax1.legend(loc='best')

        fig.tight_layout()
        plt.grid(True)
        #plt.title(f"Résolution test pour chaque découpe de {target} vs {x_variable}")
        plt.show()

def evolution_correlation(liste_decoupe, critere_decoupe) :
    df_corr = pd.DataFrame({
        "decoupe": [f"[{decoupe[critere_decoupe].min():.2f}, {decoupe[critere_decoupe].max():.2f}]" for decoupe in liste_decoupe],
        "corr_energy_charge_totale": [decoupe['energy'].corr(decoupe['charge_totale']) for decoupe in liste_decoupe],
        "corr_energy_n_hits": [decoupe['energy'].corr(decoupe['n_hits']) for decoupe in liste_decoupe],
    })
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df_corr['decoupe'], df_corr['corr_energy_charge_totale'], color='r', marker='o', label='Corrélation energy - charge_totale')
    ax1.set_xlabel('Découpe')
    ax1.set_ylabel('Corrélation')
    ax1.tick_params(axis='y')
    ax1.plot(df_corr['decoupe'], df_corr['corr_energy_n_hits'], color='g', marker='x', label='Corrélation energy - n_hits')
    ax1.set_ylabel('Corrélation')
    ax1.tick_params(axis='y')

    ax1.set_title(f'Évolution de la corrélation en fonction {critere_decoupe}')
    ax1.legend(loc='best')
    ax1.set_xticklabels(df_corr['decoupe'], rotation=45, ha='right')

    ax1.grid(True)
    fig.tight_layout()
    plt.show()

def print_binned_barplot_reg(
    liste_decoupe,
    critere_decoupe,
    cible,
    x_variable,
    reg=True,
    cmap="magma"
):

    ncols = 4
    nrows = len(liste_decoupe) // ncols + (len(liste_decoupe) % ncols > 0)
    figsize = (24, 4 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (decoupe, ax) in enumerate(zip(liste_decoupe, axes)):
        x = decoupe[x_variable]
        y = decoupe[cible]

        # Définir les bins (tu peux adapter nbins si tu veux)
        x_bins = np.linspace(x.min(), x.max(), 50)

        y_bins = np.linspace(y.min(), y.max(), 10)

        # Histogramme 2D avec colormap bien contrastée et échelle log
        h, xedges, yedges, im = ax.hist2d(
            x, y,
            bins=[x_bins, y_bins],
            cmap=cmap,  # ou 'plasma', 'viridis', etc.
            norm=colors.LogNorm(),  # met les faibles counts en valeur
            density=False
            )
        # Ajouter colorbar locale
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Nombre d'éléments (count)", fontsize=9)

        # Régression linéaire
        if reg and len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x_vals = np.linspace(x.min(), x.max(), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, '--', color='red')


            # Affichage des stats
            ax.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top', color='red')

            # Résolution
            res_str = 'res = N/A'
            rmse_str = 'rmse = N/A'
            if len(decoupe) > 5:
                try:
                    X_val = decoupe[[x_variable]]
                    y_val = decoupe[cible]
                    model_val = LinearRegression()
                    res_dict = validation_croisee(model_val, X_val, y_val, cv=5, verbose=False)
                    
                    if 'res_test_mean' in res_dict:
                        res_str = f"res = {res_dict['res_test_mean']:.2f}%"
                    if 'rmse_test_mean' in res_dict:
                        rmse_str = f"rmse = {res_dict['rmse_test_mean']:.2f}"
                except Exception as e:
                    print(f"Error in validation_croisee: {e}")
            
            ax.text(0.05, 0.90, res_str,
                    transform=ax.transAxes, fontsize=8, verticalalignment='top', color='red')
            ax.text(0.05, 0.80, rmse_str,
                    transform=ax.transAxes, fontsize=8, verticalalignment='top', color='red')

        # Nombre d’éléments
        ax.text(0.05, 0.85, f'Nb élém = {len(decoupe)}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top', color='red')

        # Titres et labels
        ax.set_title(f'Découpe {i+1} : {critere_decoupe} ∈ [{decoupe[critere_decoupe].min():.2f}, {decoupe[critere_decoupe].max():.2f}]', fontsize=10)
        ax.set_xlabel(x_variable)
        ax.set_ylabel(cible)
        ax.grid(True)

    # Supprimer les axes inutilisés
    for j in range(len(liste_decoupe), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{cible} vs {x_variable} (colorbar = count par bin)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
