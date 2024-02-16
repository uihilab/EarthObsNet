# EarthObsNet

### The code and data access for the paper - EarthObsNet: A Comprehensive Dataset for Data-Driven Earth Observation Image Synthesis

EarthObsNet contains ready-to-analysis image data for several research activities including:<br/> 
a) Earth observation image synthesis using physical driving forces and previous Earth observation images and<br/> 
b) Independent semantic segmentation study and downstream studies of a).

#### Our previous works proposed three approaches (S0, SA, and SS) to address objective a). 

* The code for those three approaches can be found in code -> mergeInput.
In particular, SA blends neighboring input with localized information through spatial aggregation, SS blends through spatial-sequence processing, whereas S0 only works on localized information (no blending).

* The code for the independent localized approach (same as S0 but with a larger dataset) can be found in code -> local. The reason the comparison of S0, SA, and SS works with a part of the dataset can be found in [this previous work](https://doi.org/10.1016/j.isprsjprs.2023.11.021). 

* The code for semantic segmentation (both for independent investigation and for downstream exploration of the results of a)) can be found in code -> binary.


#### The data is stored with Amazon S3 service. The S3 URL is <ins>s3://earthobsnet/data/</ins>.<br/>
The dataset contains 34 SAR images and a few meteorological and surface input layers with the same spatial scope as those images. The data was collected during the [2019 Central US Floods](https://appliedsciences.nasa.gov/what-we-do/disasters/disasters-activations/central-us-flooding-and-storms-spring-2019). Data from different sources are pre-processed, stacked, aligned at the pixel-level, and sliced into non-overlapping 256-by-256-pixel patches.  

The dataset is organized following the 34 SAR images' IDs (identified by the four-digit string of each zip file). Each ID will have a **_core_xxxx.zip_** file, such as _core_086E.zip_ and _core_344F.zip_. The core file is about 0.8 GB each and contains all the data needed for both activities a) and b) with the default experiment setting as introduced in our previous works. As default, neighboring image patches within 5 hops will be considered and three data options to aggregate those neighboring information - two-step averaging (inside the aggInput_avgavg folder), two-step sum (inside the aggInput folder), and two-sep maximizing (inside the aggInput_maxmax folder) are provided. We highly recommend readers check our [previous work](https://doi.org/10.1016/j.isprsjprs.2023.11.021) for more technical details regarding the neighboring image patch selection and data aggregation.  

#### In addition to the core dataset for each image ID, our dataset allows users to generate custom sub-datasets.
Suppose you want to customize neighboring patches (i.e., less or more than 5-hop) and apply a new data aggregation approach for those neighboring patches. In that case, you can do so using the complete dataset with any specific ID without **_core__** in the file name. Note that the complete dataset for each image ID is large and may contain 2 to 3 separate zip files. For instance, AC44.z01, AC44.z02, and AC44.zip are different parts of the complete dataset of the AC44 image. Inside each zip file, you will find image patches (inside the _slicing_ folder) and the corresponding DEM files (inside the _slicing_DEM_folder).  

#### Note
* Band sequence for image patches (those inside both the _slicing_ folder of the complete dataset and the _Tr_p10_ folder of the core dataset for each image ID):<br/>
Band 1 - target SAR image. <br/>
Band 2 - HAND. <br/>
Band 3 - land cover. <br/>
Band 4 - cumulative precipitation over 24 h (0-1 day's rainfall before the target time). <br/>
Band 5 - cumulative precipitation over 25-48 h (1-2 day's rainfall before the target time). <br/>
Band 6 - cumulative precipitation over 49-72 h (2-3 day's rainfall before the target time). <br/>
Band 7 - cumulative precipitation over 73-120 h (3-5 day's rainfall before the target time). <br/>
Band 8 - cumulative precipitation over 121-168 h (5-7 day's rainfall before the target time). <br/>
Band 9 - soil moisture obtained during the 3-7 days before the target date. <br/>
Band 10 - soil moisture obtained during the first 3 days before the target date. <br/>
Band 11 - SAR image captured 12 days before the target date. <br/>

* The files inside the _labels_ folder of each core zip file are the flood extent images that can be used as labels for semantic segmentation tests.
* The files inside the _slope_ folder of each core zip file  can be used as independent variables for semantic segmentation tests.
* The _Tr_p10_ folder of each core zip file means each image patch inside has more than 10 % of the total area being pre-classified as open water as pre-classified and indicated by the land cover layer. This is because hillslope pixels that are too far away from the river channel or lake area may not change significantly during rainfall and flooding processes.<br/>

If needed, users can generate customized sub-datasets with different thresholds for open water areas using the complete dataset for each image ID. 



