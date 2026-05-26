# Modélisation du risque de crédit Intel (INTC) avec le modèle de Merton

Ce projet Python vise à modéliser le risque de crédit dIntel (INTC) à laide du modèle de Merton.
L'approche est structurée en plusieurs étapes :

- collecte de données de marché
- calibration du modèle
- calcul d'une courbe de survie
- pricing d'obligations
- simulation Monte-Carlo

## Structure du projet

- `data/` : données téléchargées et base de données locale
- `src/` : code source organisé par étape
  - `src/data_collection/`
  - `src/calibration/`
  - `src/survival_curve/`
  - `src/bond_pricing/`
  - `src/monte_carlo/`
- `outputs/` : graphiques, résultats et exports

## Installation

1. Créer un environnement virtuel Python
2. Activer l'environnement
3. Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

## Étapes prévues

1. Collecte des données de marché Intel (cours, volatilité, taux sans risque)
2. Calibration du modèle de Merton sur les données de marché
3. Calcul d'une courbe de survie implicite pour l'entreprise
4. Pricing d'obligations d'Intel via le modèle
5. Simulation Monte-Carlo des scénarios de défaut et des pertes
