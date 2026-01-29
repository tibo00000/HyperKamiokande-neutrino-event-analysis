
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_processing.utils import DataMissingError

class modele_cyclique :
    def __init__(self, df_fit, x_variables_1, x_variables_2, x_variables_towall, verbose=False):
        if verbose:
            print("Initialisation du modèle cyclique...")
            print("............................................")
            print(f"Fit du 1er modèle sur l'energie avec les variables {' | '.join(x_variables_1)}....")
        self.model_energy_1 = LinearRegression()
        self.model_energy_1.fit(df_fit[["charge_totale", "n_hits"]], df_fit["energy"])
        self.variable_energy_1 = ["charge_totale", "n_hits"]
        if verbose:
            print(".............................................")
            print(f"Fit du 2ème modèle sur towall avec les variables {' | '.join(x_variables_2)} en fonction de tranches d'energie....")
        self.bins_energy = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1001]

        if verbose : print(f"Découpe energy : {'-'.join(map(str, self.bins_energy))}")
        self.models_towall = {}
        for k in range(len(self.bins_energy)-1):
            key = (self.bins_energy[k], self.bins_energy[k+1])
            poly = PolynomialFeatures(degree=3, include_bias=False)
            lr = LinearRegression()
            self.models_towall[key] = (poly, lr)
        # On fit un modèle pour chaque tranche d'énergie
        for (min_bound, max_bound), (poly, lr) in self.models_towall.items():
            df_fit_bound = df_fit[(df_fit["energy"] >= min_bound) & (df_fit["energy"] < max_bound)]
            if len(df_fit_bound) > 300:
                # Appliquer la transformation polynomiale sur les features
                X_poly = poly.fit_transform(df_fit_bound[x_variables_towall])
                y = df_fit_bound["towall"]
                # Fit du modèle linéaire sur les features transformées
                lr.fit(X_poly, y)
                if verbose :
                    print(f"Fit (sur {len(df_fit_bound)} données) modèle towall pour {min_bound}-{max_bound} MeV : \n ...")
            else:
                print(f"Aucune donnée pour la tranche {min_bound}-{max_bound} MeV, modèle non entraîné. \n ...")
                raise DataMissingError(f"Aucune donnée pour la tranche {min_bound}-{max_bound} MeV, modèle non entraîné.")
        self.variable_towall = x_variables_towall
        if verbose:
            print(".............................................")
            print(f"Fit du 2ème modèle sur l'energie avec les variables {' | '.join(x_variables_2)} en fonction du towall prédit....")
        self.bins_towall = [0, 250, 500, 1000, 3000, 5000, 9000]
        self.models_energy_2 = { (self.bins_towall[k], self.bins_towall[k+1]) : LinearRegression() for k in range(len(self.bins_towall)-1)} 
        # On fit un modèle pour chaque tranche de towall
        for (min_bound, max_bound), model in self.models_energy_2.items():
            df_fit_bound = df_fit[(df_fit["towall"] >= min_bound) & (df_fit["towall"] < max_bound)]
            if len(df_fit_bound) > 300:
                model.fit(df_fit_bound[x_variables_2], df_fit_bound["energy"])
                if verbose :
                    print(f"Fit (sur {len(df_fit_bound)} données) modèle energy pour {min_bound}-{max_bound} cm : \n ...")
            else:
                print(f"{len(df_fit_bound)} données pour la tranche {min_bound}-{max_bound} cm, modèle non entraîné (<300). \n ...")
                raise DataMissingError(f"Aucune donnée pour la tranche {min_bound}-{max_bound} cm, modèle non entraîné.")
        self.variable_energy_2 = x_variables_2
        
    
    def predict(self, df, indexes=None):
        if indexes is None:
            indexes = df.index

        df_copy = df.copy()

        # 1) Prédiction initiale de l'énergie
        y_pred_energy_1 = self.model_energy_1.predict(df_copy.loc[indexes, self.variable_energy_1])
        df_copy.loc[indexes, 'energy_pred_1'] = y_pred_energy_1

        # 2) Prédiction de towall par tranche d’énergie
        y_pred_towall = []

        for i, idx in enumerate(indexes):
            energy_pred = y_pred_energy_1[i]

            if not (self.bins_energy[0] <= energy_pred < self.bins_energy[-1]):
                energy_pred = np.clip(energy_pred, self.bins_energy[0], self.bins_energy[-1] - 1)
            # Trouver l’intervalle d’énergie correspondant
            selected_model = None
            for (min_bound, max_bound), (poly, lr) in self.models_towall.items():
                if min_bound <= energy_pred < max_bound:
                    selected_model = (poly, lr)
                    break

            if selected_model is None:
                raise ValueError(f"Énergie prédite {energy_pred:.2f} MeV hors des bornes définies.")

            poly, lr = selected_model
            
            X_row = df_copy.loc[[idx], self.variable_towall]
            X_poly = poly.transform(X_row)
            tow = lr.predict(X_poly)[0]
            y_pred_towall.append(tow)

        df_copy.loc[indexes, 'towall_pred'] = y_pred_towall

        # 3) Prédiction finale d’énergie avec les modèles par tranche de towall
        y_pred_energy_2 = []

        for i, idx in enumerate(indexes):
            towall_pred = y_pred_towall[i]
            
            if not (self.bins_towall[0] <= towall_pred < self.bins_towall[-1]):
                towall_pred = np.clip(towall_pred, self.bins_towall[0], self.bins_towall[-1] - 1)

            selected_model = None
            for (min_bound, max_bound), model in self.models_energy_2.items():
                if min_bound <= towall_pred < max_bound:
                    selected_model = model
                    break

            if selected_model is None:
                raise ValueError(f"towall prédit {towall_pred:.2f} hors des bornes définies.")

            X_row = df_copy.loc[[idx], self.variable_energy_2]
            energy2 = np.clip(selected_model.predict(X_row)[0], self.bins_energy[0], self.bins_energy[-1] - 1)
            # if not (self.bins_energy[0] - 100 <= energy2 < self.bins_energy[-1] + 100):
            #     print(f"Énergie prédite {energy2:.2f} MeV hors des bornes définies.")
            #     print("Towall prédit : ", towall_pred)
            #     print("infos event : ", df_copy.loc[idx])
            #     #clip l'énergie prédite pour éviter les erreurs
            #     energy2 = y_pred_energy_1[i]  # Utiliser l'énergie prédite initialement si hors bornes
            y_pred_energy_2.append(energy2)

        return np.array(y_pred_energy_2), np.array(y_pred_towall), np.array(y_pred_energy_1)
