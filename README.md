# EarthObsNet
The code and data access for the paper - EarthObsNet: A Comprehensive Dataset for Data-Driven Earth Observation Image Synthesis

EarthObsNet contains ready-to-analysis image data for several research activities including: 
a) Earth observation image synthesis using physical driving forces and previous Earth observation images and 
b) Independent semantic segmentation study and downstream studies of a).

Our previous works proposed three approaches (S0, SA, and SS) to address objective a). The code can be found in code -> mergeInput.
In particular, SA blends neighboring input with localized information through spatial aggregation, SS blends through spatial-sequence processing, whereas S0 only works on localized information (no blending).

The code for the independent localized approach (same as S0 but with a larger dataset) can be found in code -> local. The reason the comparison of S0, SA, and SS requires a particular part of the dataset can be found in [this previous work](https://doi.org/10.1016/j.isprsjprs.2023.11.021). 

The code for semantic segmentation (both for independent investigation and for downstream exploration of the results of a)) can be found in code -> binary.


The data is stored with Amazon S3 service. The dataset involves spatial scopes covered by 34 SAR images during the 2019 Central US Floods. Data from different sources, such as precipitation, land cover, and HAND, are stacked and sliced into non-overlapping patches.  

The dataset is organized following the 34 SAR image IDs (identified by the four-digit string of each zip file). For each ID, there will be a _core.zip file which is about 0.8 GB each and contains all data you will need for both activities a) and b). As default, neighboring image patches within 5 hops will be considered and three data aggregation options - two-step averaging, two-step sum, and two-sep maximizing are provided. We highly recommend readers check our [previous work](https://doi.org/10.1016/j.isprsjprs.2023.11.021) for more technical details.  

If you want to generate your neighboring patches (i.e., less or more than 5-hop) and apply a new data aggregation approach for those patches, you can do so using the complete dataset with any specific ID without _core. Please note the complete dataset is large and may contain 2 to 3 separate zip files so that there will not be a single but super large zip file. Inside each of those zip files, you will find image patches and the corresponding DEM file with the same name.  


