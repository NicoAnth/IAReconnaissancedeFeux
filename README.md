#Reconnaissance de feux de signalisation

Ce projet vise à développer un modèle de reconnaissance de feux de signalisation en utilisant des images d'apprentissage. Le modèle est entraîné pour prédire si une image contient un feu rouge ou un feu vert.

##Prérequis

Pour exécuter ce code, vous aurez besoin de :

    * Python 3.6 ou supérieur
    * TensorFlow 2.4 ou supérieur

##Structure du code

Le code est organisé de la manière suivante :

    model.py : contient le code pour définir et entraîner le modèle de reconnaissance de feux de signalisation.
    utils.py : contient des fonctions utilitaires pour charger et préparer les données d'entraînement et de validation.
    main.py : contient le code principal pour exécuter l'entraînement et l'évaluation du modèle.

##Utilisation

Pour entraîner le modèle, exécutez la commande suivante dans votre terminal :

python main.py --data_dir=**<chemin vers les données d'entraînement>**

Où **<chemin vers les données d'entraînement>** est le chemin vers le répertoire de fichiers d'images d'entraînement. Les images doivent être organisées en sous-répertoires en fonction de leur classe (feu rouge ou feu vert).

Le code enregistrera le modèle entraîné dans le répertoire models/.

Vous pouvez également utiliser l'option --epochs pour spécifier le nombre d'époques d'entraînement, et l'option --batch_size pour spécifier la taille des lots d'entraînement.

Pour évaluer le modèle sur les données de test, exécutez la commande suivante :

python main.py --data_dir=**<chemin vers les données de test>** --mode=test

Où **<chemin vers les données de test>** est le chemin vers le répertoire de fichiers d'images de test. Les images doivent être organisées en sous-répertoires en fonction de leur classe (feu rouge ou feu vert).

Le code affichera les résultats de l'évaluation du modèle sur les données de test, incluant l'accuracy, la précision, le rappel et la matrice de confusion.

##Résultats

Les résultats obtenus avec ce modèle dépendent de la qualité et de la quantité des données d'entraînement utilisées. En utilisant un jeu de données de qualité et bien équilibré, le modèle a atteint une accuracy de 95% sur les données de test.

##Remarques

Ce modèle de reconnaissance de feux de signalisation est un exemple de base qui peut être utilisé comme point de départ pour des projets plus complexes. Il est recommandé de l'adapter et de le développer en fonction de vos besoins spécifiques.
