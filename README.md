# EarthObsNet
The code and data access for the paper - EarthObsNet: A Comprehensive Dataset for Data-Driven Earth Observation Image Synthesis

EarthObsNet contains ready-to-analysis image data for several research activities including: 
a) Earth observation image synthesis using physical driving forces and previous Earth observation images and 
b) Independent semantic segmentation study and downstream studies of a).

Our previous works proposed three approaches (S0, SA, and SS) to address objective a). The code can be found in code -> mergeInput.
In particular, SA blends neighboring input with localized information through spatial aggregation, SS blends through spatial-sequence processing, whereas S0 only works on localized information (no blending).

The code for the independent localized approach (same as S0 but with a larger dataset) can be found in code -> local. The reason the comparison of S0, SA, and SS requires a particular part of the dataset can be found in [this previous work] (https://doi.org/10.1016/j.isprsjprs.2023.11.021). 

