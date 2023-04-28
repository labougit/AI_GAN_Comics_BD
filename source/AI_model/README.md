# AI_GAN_Comics_BD
## Comparaison modèle entre le Pix2Pix et le GAN

### GAN :

- Modèle non supervisé
- Utilise deux réseaux neuronaux : Générateur et Discriminateur
- Le générateur crée des images à partir de donnés d'entrainement
- Le discriminateur évalue la qualité de l'image générée en la comparant à des images réelles
- Les deux réseaux sont entrainés simultanément
- La création des colors en sketch se fait par un réseaux de neurone CNN

### PIX2PIX : 

- Modèle supervisé
- Utilise un ensemble de données en noir et blanc et leur version en couleur correspondantes
- Apprend à cartographier les pixels noir et blanc
- Généralise cette cartographie (décodage) pour colorier des images en noirs et blanc à couleur
- La création des colors en sketch se fait par du traitement d'image avec OpenCV
