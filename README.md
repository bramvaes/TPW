# **Slow true polar wander around varying equatorial axes since 320 Ma**

### **by B. Vaes & D.J.J. van Hinsbergen**

This manuscript was published in *AGU Advances* in 2025:

Vaes, B. & van Hinsbergen, D. J. J. (2025). Slow true polar wander around varying equatorial axes since 320 Ma. AGU Advances.

-------
Repo for the data files and Python codes used to compute the magnitude, rate, and direction of true polar wander (TPW) during the last 320 Ma.

The Jupyter Notebook named TPW.ipynb was used to perform to calculations and generate the figures of the paper.
To use the notebook, the following accompanying files are needed:
- GAPWAP_plate_circuit_*.rot: GPlates rotation file that contains the global plate circuit of Vaes et al. (2023, Earth-Science Reviews) in one of the four mantle reference frames used in this study.
- rot_functions.py: Python code with functions needed for the computation of the TPW paths
- input_APWP.xlsx: Excel file with the global APWPs used as input for the construction of the TPW paths
- rates_T14.csv: CSV file with the net TPW angles and rates of Torsvik et al. (2014, PNAS)
