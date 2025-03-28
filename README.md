

<a name="readme-top"></a>
<!--





<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique">
    <img src="images/logo_safe.jpg" alt="Logo" width=600>
  </a>

<h3 align="center">  Research Project Template </h3>

  <p align="center">
     Fast train with hydra and lightning
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique"><strong>Explore the docs »</strong></a>

  </p>
</div>


install xformers 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124


# Usage
Installe le readme
Run main.py ou template.ipynb( non recommandé)pour avoir les résultats

# Clearml
[lien](https://app.clear.ml/projects/60a7eab5e977445a83c87292650c5205/tasks?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&columns=m.e19b98ee105b15a49c56799b5e899828.e19b98ee105b15a49c56799b5e899828.value.val_accuracy.val_accuracy&columns=m.40d27a3a37f4b39e15f13b31efce253c.40d27a3a37f4b39e15f13b31efce253c.value.val_loss.val_loss&order=-last_update&filter=
)
# Lien article 
https://plmlatex.math.cnrs.fr/6883838787kyvnjgwzxsns


# A implémenter:
-> Sélection par rapport aux cliniques/ nom des cliniques . ( en trainig/test)
-> Regarder si il n'y a pas un biais sur les données? 
-> Sélection par rapport à différent backbone ( dino v2/dino large)
selon GPT:
  -> VGG-16
  -> MobileNetV3
  -> Resnet-18/50
  -> Swin-Transformer 
  -> ResNest (????)

-> Changer le finetune layer
  -> Ajouter LoRA? 
  ->  Ajouter Un linear probing plus complexe(?)
  -> 
-> Train un modèle basique à partir de rien (CNN/Transformers)
-> Ajouter de la data augmentation 
  -> CutMIX
  -> Noise/flip/Distort? ( en backbone ou en Linear Probing)

-> Rajouter plus de métriques pour le rapport?

-> Self supervised learning?