# Analyzing the complexity of ice with explainable machine learning for the development of an ice material model
# Software codes

### Introduction
In this repository you can find the software codes I used and programmed for my PhD thesis. Feel free to reuse and distribute as long as you respect and adhere to the licence requirements. At the time of uploading (21st of December, 2021) everything is working. No future updates or maintenance will be done. If you have any questions, feel free to let me know: leon.kellner@tuhh.de. 

### Corresponding publications
<ol>
  <li> L. Kellner, “Analyzing the complexity of ice with explainable machine learning for the development of an ice material model,” PhD thesis, Hamburg University of Technology, Hamburg, Germany, 2022. </li>
  <li> L. Kellner et al., “Establishing a common database of ice experiments and using machine learning to understand and predict ice behavior,” Cold Regions Science and Technology, 2019, doi: 10.1016/j.coldregions.2019.02.007. </li>
</ol>

### Explainable machine learning analysis
The data you need to run these codes is generally available but not public. Write to me (mail address above) or to Sören Ehlers (ehlers@tuhh.de) for the data. Alternatively, somebody at our institute should be able to assist you (https://www2.tuhh.de/skf/). The corresponding chapters in my PhD thesis are 5 and 6.

The following code files were part of this analysis:
1. `behavior_XGBoost.ipynb`
2. `behavior_ANN.ipynb`
3. `strength_XGBoost.ipynb`
4. `strength_ANN.ipynb`
5. `auxiliary_functions.py`
6. `data_preprocessing.py`
7. `strength_empirical.py`
8. `behavior_analytical.py`
9. `exploratory_all_data.py`
10. `exploratory_strength_values.py`
11. `model_performance.py`

The jupyter notebooks (1-4) are for training the models and for doing the XAI analyses. The python files 5 and 6 include help functions and any data preprocessing. The python files 7-11 include exploratory analyses, postprocessing, and non-machine learning models I used.

### RVE study
The corresponding chapters in my PhD thesis are  4, 7 and 8.

The following code files were part of this analysis:
1. `neper_to_hypermesh.py`
2. `create_anisotropic_grains.py`
3. `create_anisotropic_grains_RVEstudy.py`
4. `tensor_manipulation.py`
5. `files_processing.py`
6. `evaluate_average_poissons_youngs_modulus.py`
7. `stress_study.py`

Briefly, the workflow is as follows: You generate a geometry with Neper to obtain a .tess file. You process this .tess file with `neper_to_hypermesh.py`. This will output a .tcl file in the tickle language, which is to be run in the meshing software HyperMesh. Then, mesh the geometry in HyperMesh and save your model as a LS-Dyna keyfile. The scripts 2 or 3 will manipulate the keyfile such that every grain part is assigned a randomly rotated elasticity tensor. It will make use of scripts 4 and 5. Note that in script 4 there are some codes adapted from https://github.com/libAtoms/matscipy. Lastly, scripts 6 and 7 were used for postprocessing results. 

You can find more information on the polycrystal modeling processes here: https://github.com/leon-dae/polycrystal_modeling.

