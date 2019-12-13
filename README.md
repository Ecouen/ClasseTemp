# AutoML Class

## Le minimum de votre class AutoML  :
### Prétraitement: 
   reduction de la taille de votre dataset (float64 => float32, object => category etc.)
   - sur les variables catégoriques :
        - gestion des nan
        - label encoder
		
   - sur les variables numériques:
        - gestion des nan
        - normalisation
### Prediction :
   - détecter automatiquement si c'est un problème de classification ou de régression
   - en fonction, utiliser les algos appropriés
   -  ressortir les performances du modèle

### save_ypred('test.csv') => sauvegarde le fichier de prediction prêt à être soumis sur kaggle
