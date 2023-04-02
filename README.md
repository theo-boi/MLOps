Démonstration d'MLOps
=========
*Supervisé par Nicolas RAGOT et conduit par Théo BOISSEAU à l'Ecole Polytechnique de l'Université de Tours.*


Objectifs
---------
Ce projet dont le sujet a été librement choisi par l’étudiant vise à compléter son enseignement sur le Machine Learning abordé de façon universitaire lors de son cursus par une sensibilisation aux besoins spécifiques de l’industrie et du monde de l’entreprise.

L’étudiant doit donc s’approprier de nouvelles connaissances et de nouveaux outils sur l’MLOps et mûrir sa réflexion sur les enjeux de tels projets.

L’objectif principal de ce projet est de s’inspirer d’un projet de Machine Learning habituel en université pour proposer une démarche permettant de livrer rapidement et de façon fiable des modèles de Machine Learning en production à travers les concepts clés de l’MLOps.

Organisation
---------
Les exigences principales de ce projet ont été bien définies, claires et partagées en amont :
- Une phase **Data Science** lors de laquelle un projet de Data Science classique issu d’un Jupyter Notebook conçu par l’étudiant ou adapté d’une plateforme de concours de Machine Learning telle que Kaggle devra être validé et simplifié pour entraîner un réseau de neurones sur le dataset d’images de nombres MNIST.
- Une phase **Cloud** lors de laquelle le code de ce Notebook devra alors être repris pour être utilisé dans le Cloud sur Azure Machine Learning et découvrir la plateforme.
- Une phase **Production** lors de laquelle le modèle entraîné devra être déployé dans un environnement de production selon les bonnes pratiques de l’MLOps : automatisation, collaboration, monitoring et évolutivité.


Prérequis
------------
L'utilisateur doit avoir Conda installé pour gérer ses packages et une souscription Azure Machine Learning personnelle ou scolaire.
Il peut utiliser une machine sur Windows ou Linux.


Installation
------------
Pour être exécuté, tout le dossier *data_science* doit être importé dans l'onglet *Notebooks* de la partie *Authoring* de Azure Machine Learning.

En ce qui concerne la partie principale du projet, **Production**, il faut installer l'environnement `production\model\conda.yaml` avec la commande:

    conda env create --name mlops --file production\model\conda.yaml
    source activate mlops


Structure du projet
-------------------

    mlops-demo
    ├── data_science/                   # code pour la phase de data science
    │   ├── input/                      # données d'entrée
    │   │   ├── test.csv                # données de test
    │   │   └── train.csv               # données d'entraînement
    │   ├── working/                    # code de travail
    │   │   ├── data_science_digits_model.ipynb    # carnet Jupyter du modèle de data science
    │   │   └── main.py                 # script principal
    │   └── cloud_AML_digits_model.ipynb  # carnet Jupyter pour le déploiement sur le cloud
    │
    ├── production/                     # code pour la phase de production
    │   ├── foreign_data/               # données externes
    │   │   ├── test.csv                # données de test
    │   │   └── train.csv               # données d'entraînement
    │   ├── model/                      # modèle entrainé issu de Azure Machine Learning
    │   │   ├── data/                   # données de modèle
    │   │   │   ├── model/              # modèle sérialisé
    │   │   │   │   ├── variables/      # variables du modèle
    │   │   │   │   ├── keras_metadata.pb  # fichier de métadonnées Keras
    │   │   │   │   └── saved_model.pb  # fichier de modèle sauvegardé
    │   │   │   ├── keras_module.txt    # fichier texte de module Keras
    │   │   │   └── save_format.txt     # fichier texte du format de sauvegarde
    │   │   ├── MLmodel                 # modèle MLFlow
    │   │   ├── _summary.txt            # résumé du modèle
    │   │   ├── conda.yaml              # environnement Conda
    │   │   ├── python_env.yaml         # environnement Python
    │   │   └── requirements.txt        # dépendances Python
    │   │
    │   ├── working/                    # code de travail
    │   │   ├── data_simulation.py      # script de simulation de données
    │   │   ├── dummy_server.py         # script du serveur factice
    │   │   ├── score.py                # script de scoring
    │   │   └── visualization.py        # script de visualisation
    │   └── production.ipynb            # carnet Jupyter pour la phase de production
    │
    ├── .gitignore                      # fichiers à ignorer lors de la confirmation dans Git
    └── README.md

-   `data_science/` contient le code des phases **Data Science** et **Cloud**.
    Le dossier *input* contient les données du dataset MNIST.
    Le dossier *working* contient le Notebook d’un projet classique de Data Science adapté aux besoins du projet.
    Le Notebook *cloud_AML_digits_model.ipynb* contient les instructions en python destinées à Azure Machine Learning et le fichier *main.py* dans le dossier *working* correspond au code qu’il a généré et déployé sur la plateforme.
    Pour le cloud, tout le dossier *data_science* doit être envoyé sur Azure Machine Learning.

-   `production/` contient le code de la phase **Production**.
    Le dossier *foreign_data* contient deux datasets étrangers au dataset MNIST occidental : un de nombres écrits par des indiens et des népalais, et un autre écrit par des japonais.
    Le dossier *model* est la sortie du job d’entraînement d’Azure Machine Learning : un artifact contenant l’environnement nécessaire pour faire fonctionner le modèle, le modèle sérialisé et des méta-données.
    Le fichier *production.ipynb* est un Notebook dans lequel est simulé un environnement de production et qui présente un exemple de déploiement du modèle.
    Dans *working* est le fichier *score.py* (un script de scoring) utilisé pour l’inférence et généré par le Notebook *production.ipynb*, ainsi que le reste du code utilisé pour simuler la production.
