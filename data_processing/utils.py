
import numpy as np
import pandas as pd

class DataMissingError(Exception):
    """Exception raised when there is missing data for a specific range."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def get_single_event_df(df, num_event):
    """
    Extracts a singe event's hits and general info into a DataFrame.
    """
    # Extraire les infos brutes
    ligne = df.iloc[num_event][["hitx", "particleDir","particleDir_x","particleDir_y",
                                "particleDir_z", "hity", "hitz", "energy", "towall", 
                                "charge_totale", "charge", "time", "dwall", "vertex_x", 
                                "vertex_y","vertex_z"]]  # Une Series
    n_hits = len(ligne["hitx"])  # Supposé être égal pour hitx, hity, hitz

    # Colonnes à répéter (infos générales de l'événement)
    cols_event = ["particleDir", "particleDir_x", "particleDir_y", "particleDir_z",
                  "energy", "towall", "charge_totale", "dwall", 
                  "vertex_x", "vertex_y","vertex_z"]

    # On construit un dictionnaire pour créer le nouveau DataFrame
    data = {
        "hitx": np.asarray(ligne["hitx"]),
        "hity": np.asarray(ligne["hity"]),
        "hitz": np.asarray(ligne["hitz"]),
        "charge" : np.asarray(ligne["charge"] ),
        "time" : np.asarray(ligne["time"] )

    }

    # Ajouter les données événementielles répétées
    for col in cols_event:
        data[col] = [ligne[col]] * n_hits

    # Finaliser le sous-DataFrame
    single_event = pd.DataFrame(data)
    # single_event.describe() # Removed print/return of describe to keep function pure
    return single_event

def get_single_event_row(row_dict):
    """
    Similaire à get_single_event_df mais prend un dictionnaire (ou row) en entrée.
    """
    # row_dict est un dictionnaire, pas une Series
    ligne = row_dict  # on renomme simplement

    n_hits = len(ligne["hitx"])  # Supposé égal pour hitx, hity, hitz

    # Colonnes à répéter
    cols_event = ["particleDir", "particleDir_x", "particleDir_y", "particleDir_z",
                  "energy", "towall", "charge_totale", "dwall", 
                  "vertex_x", "vertex_y","vertex_z"]

    # On construit les données hits
    data = {
        "hitx": np.asarray(ligne["hitx"]),
        "hity": np.asarray(ligne["hity"]),
        "hitz": np.asarray(ligne["hitz"]),
        "charge": np.asarray(ligne["charge"]),
        "time": np.asarray(ligne["time"]),
    }

    # Ajouter les données de l’événement répétées pour chaque hit
    for col in cols_event:
        data[col] = [ligne[col]] * n_hits

    # Créer le DataFrame final de l'événement
    single_event = pd.DataFrame(data)
    return single_event

def trouver_angle_pmt_particle_dir(pmt_pos, particleDir, HK, seuil=1e-2):
    """
    Calcule l'angle d'incidence

    :param pmt_pos: [x, y, z] position du PMT
    :param particleDir : [x,y,z] direction de la particule
    :param HK: dictionnaire contenant 'height', 'cylinder_radius' et 'maxi' (hauteur max)
    :param seuil: indique si PMT aux extrémité du cylindre
    :return: angle en degrés entre le vecteur lumière et le plan tangent
    """
    x0, y0, z0 = pmt_pos

    # Vecteur lumière
    v = np.asarray(particleDir)
    v_norm = np.linalg.norm(v)

    #vecteur normal selon la position du pmt (disque ou coté)
    if abs(z0) < HK["maxi"] - seuil :
        n = np.array([x0, y0, 0])
    else :
        n = np.array([0, 0, z0])
    n_norm = np.linalg.norm(n)

    # Produit scalaire
    dot = np.dot(v, n)

    # Angle entre lumière et normale
    theta_rad = np.arccos(dot / (v_norm * n_norm))
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg if  theta_deg< 90 else 180 - theta_deg

def corriger_charge_angle(charge, angle):
    """Corrige la charge capté en fonction de l'angle d'incidence

    Args:
        charge
        angle d'incidence

    Returns:
        charge corrigée
    """
    angle = np.clip(angle, 0, 90)
    angle_rad = np.radians(angle)  
    return charge / np.clip(np.cos(angle_rad), 0.2, 1)

def corriger_absorption(charge, kappa, distance):
    return charge * np.exp(distance / kappa)

def corriger_charge(charge, kappa, distance, angle):
    charge_corrigee_absorption = corriger_absorption(charge, kappa, distance)
    #charge_corrigee = charge_corrigee_absorption
    charge_corrigee = corriger_charge_angle(charge_corrigee_absorption, angle)
    return charge_corrigee

def get_charge_totale_corr(row):
    event = get_single_event_row(row)
    
    # 1. Calcul vectorisé de la distance vertex -> hit
    dx = event["hitx"] - event["vertex_x"]
    dy = event["hity"] - event["vertex_y"]
    dz = event["hitz"] - event["vertex_z"]
    event["distance_vertex_to_hit"] = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 2. Calcul vectorisé de l’angle d’incidence
    HK = {
        'height': 6575.1,
        'cylinder_radius': 6480 / 2,
        'PMT_radius': 25.4,
        'maxi': 3296.4712
    }
    # Si trouver_angle_pmt_particle_dir n'est pas vectorisée, on peut faire un apply UNE seule fois ici :
    event["angle_hit"] = event.apply(
        lambda r: trouver_angle_pmt_particle_dir(
            pmt_pos=[r["hitx"], r["hity"], r["hitz"]],
            particleDir=r["particleDir"],
            HK=HK
        ), axis=1
    )
    
    # 3. Correction absorption (vectorisé)
    kappa = 1200
    charge_corr_abs = event["charge"] * np.exp(event["distance_vertex_to_hit"] / kappa)
    
    # 4. Correction angle (vectorisé)
    angle_rad = np.radians(np.clip(event["angle_hit"], 0, 90))
    cos_theta = np.clip(np.cos(angle_rad), 0.2, 1)
    charge_corr_angle = event["charge"] / cos_theta
    
    # 5. Correction combinée (vectorisé)
    charge_corr_finale = charge_corr_abs / cos_theta

    # 6. Totaux
    total_corr_abs = np.sum(charge_corr_abs)
    total_corr_angle = np.sum(charge_corr_angle)
    total_corr_finale = np.sum(charge_corr_finale)

    return total_corr_finale, total_corr_angle, total_corr_abs
