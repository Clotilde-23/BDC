# Housing price prediction
By Louise Bonhomme, Michel Daher Mansour, Antonin Falher, Blanche Lalouette and Clotilde Nietge.

This project is realised in the context of the Buisness Data Challenge at Ensae. The partner in this project is MeilleurTaux a reference market place for loans, insurance and investments and offers various services for individuals and professionals: financing (real estate loans,loan consolidation, consumer credit, professional credit), insurance (home, and loan insurance)
and investments. 

In this work, we focused on four cities: Paris, Lyon, Marseille and Toulouse.
We managed to predict prices with a median absolute percentage error of 7.93\% for Paris, 8.71\% for Lyon, 9.84\% for Toulouse.


## Structure of the repository

* Notebooks : You can find all the notebooks in this file. For each city, there are the results of the models. The Notebook ```Dataset_Cleaning and processing.ipynb``` has cleaned the whole "Demande de Valeurs Foncières" dataset to have smaller datasets that we can use. Those datasets have been saved in the file __Data__.

* Data : You can find datasets used for the models of estimating housing price (created with ```Dataset_Cleaning and processing.ipynb``` Notebook) and public datasets from Insee or metro stations to add external socio-demographic information.

* Helpers : You can find the detailed functions that we used in the Notebooks.
