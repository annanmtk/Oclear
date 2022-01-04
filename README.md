![Capture](https://user-images.githubusercontent.com/88719925/148106824-148d1bf1-937c-4171-84d7-4aad7102f5a5.PNG)
# Oclear
Oclear est un projet de lecture automatique des chèques
c’est à dire les informations écrites par l'utilisateur, y compris
la date, la signature, le montant en chiffre, le montant en
lettre et le nom du bénéficier.
Chacun des champs ci-dessus a une position bien précise
dans le chèque comme vous pouvez le voir dans l’image cidessus.
Ces informations sont vérifiées visuellement par un caissier à
chaque retrait de cheque par un client. L’objectif de ce projet
est de pouvoir lire ces informations automatiquement à
travers une application web.
C’est à dire, de construire un(des) modele(s) de Machine
Learning (Optical character recognition) capable de
reconnaitre les écritures manuscrites qui sont dans un
chèque. Cet model sera intègre dans une application mobile
et se chargera de la lecture automatique des chèques.
A la fin de ce projet, nous espérons dans un chèque, pouvoir
reconnaitre :

1. Les écritures manuscrites des chiffres (montant en
chiffre, date)
2. Les écritures manuscrites des caractères (montant en
lettre, nom du bénéficier, lieu).
3. Comparer le montant en chiffre et le montant en lettre
4. Comparer la signature du client qui dans le chèque et la
signature du même client si se trouve dans la base de
données de la banque.
5. Vérifier toutes les mentions obligatoires sur le chèque.


Aller tester sur le fichier Load_and Prediction.ipynb
