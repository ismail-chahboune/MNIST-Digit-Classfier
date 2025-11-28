# Optimal MNIST Digit Classifier using PyTorch

##  Description du Projet
Ce projet consiste à entraîner un modèle de **réseau de neurones convolutionnel optimisé** pour la classification des chiffres MNIST (0 à 9).  
Le modèle est conçu pour être **léger et performant**, avec augmentation des données et régularisation via Dropout.

---

##  Objectifs du Projet
- Charger et transformer le dataset MNIST  
- Définir un **CNN optimisé** pour les images 28x28  
- Entraîner le modèle avec **Adam optimizer** et scheduler de learning rate  
- Évaluer les performances sur le test set  
- Sauvegarder le meilleur modèle (`best_mnist_model.pth`)  
- Permettre la prédiction sur des images individuelles via `predict_digit()`

---

##  Approche
1. Chargement des datasets MNIST pour entraînement et test  
2. Augmentation des données pour le dataset d’entraînement  
3. Définition du modèle `OptimalMNISTNet` :
   - 3 couches convolutionnelles  
   - 2 couches fully connected  
   - Dropout pour régularisation  
4. Entraînement sur **10 epochs** avec suivi de la précision et du loss  
5. Sauvegarde du meilleur modèle selon la précision sur test set  
6. Fonction `predict_digit(image_tensor, model)` pour prédire un chiffre individuel  
7. Fonction `test_saved_model()` pour évaluer le modèle sauvegardé sur tout le test set  

---

##  Résultats
- Précision sur le test set après 10 epochs  
- Graphiques de loss et accuracy disponibles dans le log console  
- Modèle sauvegardé pour usage futur

---

##  Fichiers Principaux
- `mnist_classifier.py` — Script complet d’entraînement et de prédiction  
- `best_mnist_model.pth` — Modèle sauvegardé  
- Dataset MNIST téléchargé automatiquement par PyTorch (`./data`)  
- `README.md` — Documentation  

---

##  Auteur
Projet réalisé par **Chahboune Ismail**
