Skip to main content
Advertisement

SpringerOpen

Search
Get published
Explore Journals
Books
About
Login
Insights into Imaging
About
Articles
Submission Guidelines
Submit manuscript
Deep learning for opportunistic, end-to-end automated assessment of epicardial adipose tissue in pre-interventional, ECG-gated spiral computed tomography
Download PDF
Original Article
Open access
Published: 19 December 2024
Deep learning for opportunistic, end-to-end automated assessment of epicardial adipose tissue in pre-interventional, ECG-gated spiral computed tomography
Maike Theis, Laura Garajová, Babak Salam, Sebastian Nowak, Wolfgang Block, Ulrike I. Attenberger, Daniel Kütting, Julian A. Luetkens & Alois M. Sprinkart 
Insights into Imaging volume 15, Article number: 301 (2024) Cite this article

1405 Accesses

1 Citations

Metricsdetails

Abstract
Objectives
Recently, epicardial adipose tissue (EAT) assessed by CT was identified as an independent mortality predictor in patients with various cardiac diseases. Our goal was to develop a deep learning pipeline for robust automatic EAT assessment in CT.

Methods
Contrast-enhanced ECG-gated cardiac and thoraco-abdominal spiral CT imaging from 1502 patients undergoing transcatheter aortic valve replacement (TAVR) was included. Slice selection at aortic valve (AV)-level and EAT segmentation were performed manually as ground truth. For slice extraction, two approaches were compared: A regression model with a 2D convolutional neural network (CNN) and a 3D CNN utilizing reinforcement learning (RL). Performance evaluation was based on mean absolute z-deviation to the manually selected AV-level (Δz). For tissue segmentation, a 2D U-Net was trained on single-slice images at AV-level and compared to the open-source body and organ analysis (BOA) framework using Dice score. Superior methods were selected for end-to-end evaluation, where mean absolute difference (MAD) of EAT area and tissue density were compared. 95% confidence intervals (CI) were assessed for all metrics.

Results
Slice extraction using RL was slightly more precise (Δz: RL 1.8 mm (95% CI: [1.6, 2.0]), 2D CNN 2.0 mm (95% CI: [1.8, 2.3])). For EAT segmentation at AV-level, the 2D U-Net outperformed BOA significantly (Dice score: 2D U-Net 91.3% (95% CI: [90.7, 91.8]), BOA 85.6% (95% CI: [84.7, 86.5])). The end-to-end evaluation revealed high agreement between automatic and manual measurements of EAT (MAD area: 1.1 cm2 (95% CI: [1.0, 1.3]), MAD density: 2.2 Hounsfield units (95% CI: [2.0, 2.5])).

Conclusions
We propose a method for robust automatic EAT assessment in spiral CT scans enabling opportunistic evaluation in clinical routine.

Critical relevance statement
Since inflammatory changes in epicardial adipose tissue (EAT) are associated with an increased risk of cardiac diseases, automated evaluation can serve as a basis for developing automated cardiac risk assessment tools, which are essential for efficient, large-scale assessment in opportunistic settings.

Key Points
Deep learning methods for automatic assessment of epicardial adipose tissue (EAT) have great potential.

A 2-step approach with slice extraction and tissue segmentation enables robust automated evaluation of EAT.

End-to-end automation enables large-scale research on the value of EAT for outcome analysis.

Graphical Abstract

Introduction
The volume and attenuation of cardiac adipose tissue (CAT) measured by CT has recently been shown to be a predictive marker for the outcome of various cardiac diseases [1,2,3,4,5,6,7,8,9]. For the assessment of CAT, a clear separation is made between epicardial adipose tissue (EAT), defined as adipose tissue between the myocardium and the pericardium, and pericardial adipose tissue (PAT), which refers to adipose tissue outside the pericardium. In the past, it has been shown that the EAT volume in particular is associated with coronary artery disease, myocardial ischemia, myocardial infarction or major adverse cardiac events [1,2,3,4,5]. Increased EAT volume was also found to be an independent predictor of all-cause mortality in patients undergoing transcatheter aortic valve replacement (TAVR) [6, 7]. In these studies, EAT volume was assessed manually or semi-manually by contouring the pericardium and applying appropriate thresholds or fully automatically using non-open, non-free commercial software [2,3,4].

The use of deep learning (DL) methods for the automation of quantitative image analyses enables an opportunistic large-scale assessment, which has already proven its worth in body composition analysis [10,11,12,13,14,15]. Also, the automation of volumetric EAT segmentation was investigated in several studies [16,17,18]. Recently, Haubold et al presented the open-source body and organ analysis (BOA) pipeline as an extension of the “TotalSegmentator” framework, which enables automated body composition analysis, including the assessment of EAT and PAT [19,20,21].

Previous research has shown a high correlation between 3D and 2D measurements at specific anatomical positions [9, 22, 23]. Therefore, one way to reduce time-consuming annotation effort in case of manual analyses is to evaluate EAT on a single slice instead of assessing EAT in the whole volume. Salam et al demonstrated the value of single-slice CAT measurements for outcome prediction in TAVR patients and found that EAT density is a significant predictor of mortality [8]. In that study, EAT was assessed at the aortic valve (AV)-level, where the important delineation of the pericardium is reliably possible in ECG-gated CT scans, which also makes automated analyses potentially more robust compared to volumetric approaches. However, while various publications address single-slice-based assessment of body composition at the lumbar level, pipelines for highly robust automated analyses of EAT are missing so far [12, 13, 24].

To automatically assess EAT from single-slice CT scans, also slice extraction must be automated in addition to the segmentation step. In the past, different DL approaches have been used to automate this task. For automated body composition analyses at the lumbar level, the selection of a specific slice was considered a segmentation task by taking the landmark as the center of mass of an ellipsoidal segment around the vertebrae [13]. Another approach used a convolutional neural network (CNN) to directly predict the spatial offset of each axial slice to the targeted anatomical landmark [12, 24]. In addition to these classical approaches of supervised learning, promising results have been observed for reinforcement learning (RL)-based methods for the detection of anatomical landmarks [25,26,27]. RL is the third paradigm of machine learning, alongside supervised and unsupervised learning, and has so far received little attention in image analysis. In RL, an agent learns from interactions with its environment, resulting in a transition to a new state of the environment. Based on the agent’s actions, it receives feedback in the form of positive and negative rewards, which it strives to maximize over time in order to make optimal decisions [28].

In this study, we focus on the development of a single-slice-based assessment of EAT in pre-interventional, ECG-gated spiral CT by automating both required processing steps, namely slice extraction and tissue segmentation. A range of different DL approaches including RL was investigated with the objective of developing a pipeline that enables robust large-scale opportunistic analyses.

Materials and methods
Dataset
This retrospective single-center study was approved by the local ethics committee with a waiver for written informed consent. The dataset used for model development consists of patients undergoing TAVR at the University Hospital Bonn with available pre-interventional contrast-enhanced thoraco-abdominal or cardiac CT scans acquired as spiral CT between 2008 and 2020, all prospectively or retrospectively ECG-gated (diastolic phase). Additionally, the trained model was also evaluated on pre-interventional, ECG-gated (diastolic) spiral CT scans of patients receiving CT for pre-operative planning prior to heart surgery. For this evaluation, 50 patients were randomly selected from all corresponding CT scans acquired between July and August 2024.

For the EAT assessment, a single axial slice was selected at the AV-level. Epicardial tissue defined as tissue within the pericardium was manually segmented on this single-slice image and adipose tissue was identified by applying a threshold of −190 to −30 Hounsfield units (HU). Manual data annotation was performed by a radiology resident (B.S.) with 3 years of experience in cardiac imaging supervised by a board-certified radiologist specialized in cardiovascular imaging (D.K.) using the open-source software 3D Slicer (version 4.11) [29].

Slice extraction
Two different DL approaches were investigated and compared for automating the slice extraction at the AV-level: A 2D CNN referred to as baseline approach and a 3D RL model. An overview of both methods can be found in Fig. 1.

Fig. 1
figure 1
Overview of both slice extraction methods. A The baseline approach consists of a 2D convolutional neural network (CNN) using the EfficientNet-B0 architecture and predicts the offset of each axial slice to the target slice at the level of the aortic valve (AV). Predicted offsets were smoothed before the final prediction of the slice with the minimal distance to the target. B The reinforcement learning (RL) approach is based on the RL-Medical package. In this case, five agents start at five different random positions, and each agent ends with its own prediction of the 3D AV landmark position. After the application of the outlier detection to the results of the five agents, the final prediction is determined by averaging the landmark predictions of all remaining agents

Full size image
For the baseline approach, two CNNs following the EfficientNet-B0 architecture were trained to predict the spatial offset of each axial slice to the target AV-level [30]. For the first CNN, the entire dataset was rescaled to a uniform slice thickness of 10 mm to obtain an initial estimate of the actual AV position. Using these rough predictions, the dataset was cropped to the heart region and the second CNN was then trained on high-resolution heart crops to determine the exact slice position. Both CNNs were trained for 1000 epochs using L1-Loss function and AdamW optimizer with a weight decay of 0.01. The learning rate was scheduled by the OneCycle learning rate policy with a maximal learning rate of 10−4. Training was stopped after no improvement on the validation set was observed for 100 epochs. The performance of the automatic slice extraction was evaluated based on the mean absolute deviation in z-direction (Δz). In a post-processing step, the output predictions of both DL models were smoothed using a moving average filter, where the use of different window sizes was evaluated. For more information on the baseline approach including pre-processing of the data and detailed training parameters, see Supplement S1.

The investigated 3D RL model is based on the open-source software RL-Medical, developed by Alansary et al for 3D landmark detection in medical images [26, 31]. It uses RL in a multi-agent setting, where each agent is positioned within a patch of the 3D medical image considered to be the agent’s state. Agents start exploration from random locations within the image volume, aiming to locate the target through gradual patch movements. The action space, which defines the freedom of motion, allows movement in six directions in 3D Cartesian coordinates. Rewards are based on Euclidean distances (ED) to incentivize movement towards the targeted landmark position. During training, a Q-network is learned as an approximation to the Q-function predicting the next direction of movement by estimating the expected benefit for each action in a state with the goal of attaining the highest possible cumulative reward. The trained network architecture (Network3D) consists of shared convolutional layers among all agents and agent-specific fully connected layers.

Results for the landmark positions were obtained by training the Network3D model with five agents simultaneously searching for the landmark AV. The final AV position was determined by averaging the predicted agents’ end positions, with outlier detection performed on the agents’ positions before averaging. This process involved using z-scores with median absolute deviation to identify and remove outliers [32]. Performance was assessed by calculating the mean ED between the resulting prediction and the target landmark. Additionally, for comparison with respect to the baseline approach, Δz was computed.

More information about the training and details about the utilized outlier detection are described in Supplement S2.

Tissue segmentation
For automatic segmentation of EAT, a 2D U-Net model with residual units was trained on single-slice images on AV-level using the open-source MONAI python framework (version 0.9.0) [33, 34]. Training was performed for 10,000 epochs using the AdamW optimizer and sum of cross-entropy and Dice coefficient as loss function. A grid search for selection of hyperparameters was conducted to find suitable settings for learning rate (10−3, 10−4) and weight decay (10−3 to 0.5), where the learning rate was adapted for each iteration step according to the OneCycle learning rate policy. A fixed batch size of 200 was used with gradient accumulation over two batches. Training was stopped after no improvement in the validation loss was observed for 100 epochs. The performance of the different hyperparameter settings was compared on the validation set according to the Dice score on which the best model was selected and finally applied to the hold-out test set.

Additionally, the publicly available BOA pipeline was applied to the hold-out test set [19]. The resulting EAT segmentation was considered on the manually selected AV-level only and was compared with ground truth segmentations generated by the radiologist. For more information on the tissue segmentation,  see Supplement S3.

End-to-end evaluation
Finally, an end-to-end evaluation of the entire pipeline was performed by analyzing and comparing mean absolute difference (MAD) of EAT density (HU) and area (cm2) once from automated EAT segmentation at the predicted AV-level and once from manual EAT segmentation at the manually selected AV-level. Therefore, the pipeline was applied to the hold-out test set consisting of spiral CTs from patients undergoing TAVR and to the 50 randomly selected spiral CT scans from patients prior to heart surgery.

Statistical evaluation
For all performance metrics, 95% confidence intervals (CI) were determined by bootstrapping the test set with 1000 resamples. For end-to-end evaluation, Pearson correlation coefficient was calculated and a Bland-Altman analysis was performed to check for any systematic differences using the Python packages seaborn (version 0.11.2), scipy (version 1.9.1), and pyCompare (version 1.5.3).

Results
Dataset
Overall, pre-interventional CT scans from 1583 patients were available, whereby 81 scans were excluded due to artifacts (e.g., caused by metal implants). Thus, 1502 TAVR patients (mean age 80.6 ± 6.3 years, 709 (47.2%) female) were included for method development. Scans were performed on four different scanners, one dual-source CT (Siemens Somatom Force) and three single-source CTs (Philips Brilliance 64, Philips iCT 256, Philips Mx8000 IDT 16). The dataset comprises 948 contrast-enhanced thoraco-abdominal and 554 cardiac computed tomography scans. For method development, the dataset was randomly split into training, validation, and hold-out test cases (n = 1052 / 224 / 226). A subset of the data has already been used in another study investigating the correlation between inflammatory changes in EAT or PAT and cardiac risk [8]. The 50 randomly selected pre-interventional CT scans from patients undergoing heart surgery (mean age 63.9 ± 15.7 years, 18 (36.0%) female) were used as additional test set for the end-to-end evaluation. The dataset comprises 25 cardiac CT scans performed on Siemens Somatom Force (Test A) and 25 contrast-enhanced thoraco-abdominal scans performed on Siemens Naeotom Alpha (Test B). For information on image characteristics, see Table 1.

Table 1 Detailed overview of image characteristics of the dataset, including pixel spacing, slice thickness, matrix size, and tube voltage
Full size table
Slice extraction
For the baseline approach, a moving average filter with a window size of 11 led to the highest performance on the validation set with Δz = 2.1 ± 1.9 mm. More details on applying the moving average filter using different window sizes can be found in Supplement S4. The application to the hold-out test set resulted in a mean deviation of 2.0 ± 2.0 mm (95% CI: [1.8, 2.3] mm).

The RL approach resulted in a mean ED of 3.4 ± 4.2 mm (max ED = 59.5 mm) and a mean Δz of 1.9 ± 3.8 mm (max Δz = 54.0 mm). The application of the outlier detection method before averaging the positions of the five agents improved the performance regarding both metrics (mean ED distance = 3.3 ± 1.9 mm, max ED = 10.0 mm; mean Δz = 1.7 ± 1.5 mm, max Δz = 7.0 mm). Applying this approach to the hold-out test set led to even smaller Δz = 1.8 ± 1.6 mm (95% CI: [1.6, 2.0] mm) compared to the baseline approach. Mean ED on the hold-out test set was 3.3 ± 2.2 mm (95% CI: [3.0, 3.6] mm). Table 2 provides a detailed comparison of both methods.

Table 2 Performance overview for both slice extraction approaches on the hold-out test set
Full size table
Tissue segmentation
Highest performance for the implemented 2D U-Net on the validation set was reached after 1512 epochs with a maximal learning rate of 10−3 and a weight decay of 0.1 (see Supplement S5). Mean Dice score was 91.6 ± 4.2%. Applying the model to the hold-out test set yielded a Dice score of 91.3 ± 4.4% (95% CI: [90.7, 91.8] %). The agreement at AV-level of the open-source pipeline BOA with the radiologist’s segmentation showed a significantly lower Dice score of 85.6 ± 7.1% (95% CI: [84.7, 86.5] %). The 2D U-Net also showed significantly higher agreement with the radiologist regarding MAD of density and area than the BOA pipeline (see Table 3).

Table 3 Comparison of the implemented 2D U-Net and the open-source body and organ analysis (BOA) pipeline with mean ± standard deviation and 95% confidence intervals in the brackets
Full size table
End-to-end evaluation
Based on the validation results, a pipeline comprising RL-based slice extraction with outlier detection and 2D U-Net-based tissue segmentation was chosen for end-to-end evaluation. An overview of the final pipeline can be found in Fig. 2. The training and inference script are accessible at https://github.com/ukb-rad-cfqiai/EAT_Assessment.

Fig. 2
figure 2
Overview of the final pipeline consisting of the reinforcement learning (RL) method for slice extraction of the aortic valve level and a 2D U-Net for segmentation of the epicardial adipose tissue

Full size image
A high agreement was observed between automated and manual assessment of EAT at the AV-level for the hold-out test set (MAD density = 2.2 ± 2.1 HU (95% CI: [2.0, 2.5] HU), MAD area = 1.1 ± 1.1 cm2 (95% CI: [1.0, 1.3] cm2)). A significant correlation was found between automatically and manually measured EAT area with a Pearson correlation coefficient of r = 0.96 (p < 0.001) and between automatically and manually measured EAT density (r = 0.89, p < 0.001). Bland-Altman analysis revealed no systematic deviation between manual and automatic end-to-end comparison, with low mean area difference of −0.02 cm2 and low mean density difference of 0.57 HU (see Fig. 3). Similar performance was observed for the additional test sets Test A and Test B consisting of pre-operative CT scans (see Table 4 and Fig. 3).

Fig. 3
figure 3
Correlation and Bland-Altman analyses for the end-to-end comparison of manually and automatically measured epicardial adipose tissue (EAT) area given in cm2 (A) and density given in Hounsfield units (HU) (B) for the hold-out test set consisting of pre-interventional TAVR patients and the test sets (Test A and Test B) consisting of pre-operative patients prior to heart surgery. The Pearson correlation coefficient (r) is provided for all analyses

Full size image
Table 4 End-to-end comparison of the final pipeline evaluated on the hold-out test set containing pre-interventional TAVR patients, the pre-operative cardiac (Test A) and the pre-operative thoraco-abdominal (Test B) scans
Full size table
Figure 4 presents examples of the end-to-end evaluation in a cardiac and a thoraco-abdominal scan. Manual segmentation time using the open-source software 3D Slicer took about 100 s per patient. In comparison, the inference time for RL-based slice extraction was about 30 s per patient for cardiac CTs and 127 s for thoraco-abdominal CT scans. Time for automatic segmentation was only 2 s per patient. Inference time was measured on an Nvidia GeForce RTX 3090 Graphics Processing Unit with 24 gigabyte video memory.

Fig. 4
figure 4
Two examples of the end-to-end comparison of manual (marked in green) and automated (marked in blue) epicardial adipose tissue (EAT) assessment for a contrast-enhanced cardiac CT (A) and an ECG-gated thoraco-abdominal CT scan (B). Δz denotes the absolute difference between manually selected and automatically predicted slice position at the level of the aortic valve

Full size image
Discussion
This paper presents an automated pipeline for opportunistic assessment of EAT from contrast-enhanced ECG-gated (diastolic) cardiac and thoraco-abdominal spiral CT scans for pre-interventional patients undergoing TAVR or heart surgery. The proposed approach essentially consists of two steps: automatic slice extraction at the AV-level and 2D segmentation of EAT.

For the slice extraction task, the RL approach achieved impressive accuracy. In contrast to the baseline approach, which aims to predict the z-distance of a slice to the targeted slice, the RL model utilizes 3D data, likely explaining its advantages over the 2D approach, which receives only a single axial slice as input. In addition, a multi-agent setting was used for the RL approach, i.e., the position of the landmark is predicted by five agents during inference. The ensembling-like averaging of the five agents after outlier detection contributed to a more robust and generalizable model. For tissue segmentation at AV-level, the 2D U-Net model achieved a higher precision for EAT segmentation compared to the volumetric segmentation approach, resulting in high agreement between manually and automatically measured area and mean density in the end-to-end evaluation.

Various approaches to automated EAT assessment have been explored in the past, ranging from traditional methods to DL algorithms to avoid time-consuming and tedious manual tasks. One approach for this purpose is atlas-based segmentation, in which the input data is registered to a set of reference images [35,36,37]. In recent years, DL methods have become established for segmentation tasks and have also already been implemented for EAT assessment. In 2019, Commandeur et al presented a multi-task DL framework for EAT segmentation in non-contrast-enhanced cardiac CTs, which takes three consecutive axial slices as input. In the first task, the DL algorithm classifies whether the central input slice is located in the heart region while in the second, the pericardium is segmented [3, 4, 16, 38]. Hoori et al proposed DeepFat, a CNN with atrous convolution for automatic segmentation of EAT in CT calcium score images. Further approaches used either generative adversarial networks or various U-Net-based architectures for segmentation [17, 18, 39,40,41]. Recently, the open-source DL model called BOA was introduced which extends classical body composition analysis to volumetric assessment of cardiac adipose tissue in CT scans [19].

In our work, we focused on automated 2D EAT assessment at the AV-level in pre-interventional CT scans of patients undergoing TAVR, as a correlation between mean EAT density measured at this single-slice and overall survival has recently been reported in this cohort [8]. In addition, we have shown that our algorithm also enables robust EAT evaluation on pre-operative CT data prior to heart surgery. A major difficulty in segmenting the entire EAT is the differentiation between pericardial and epicardial tissue, as the pericardium cannot always be clearly delineated on every CT image. Thus, significant differences have been found in the past when comparing the EAT assessment of two readers, and a high inter-reader variability of 15% has been reported for EAT volume measurement [16, 42]. In contrast, a low inter-reader variability of 0.1% was reported at the AV-level for both the measured EAT area and the mean density, suggesting that the delineation of epicardial tissue on this slice is more robust and reliable, which is very likely also true for automated opportunistic assessment [8]. This aspect probably explains why the proposed 2D approach was superior to the 3D BOA pipeline, as our method was explicitly trained at AV-level, where EAT segmentation can be performed with higher reliability. However, it remains an open question as to which method is most appropriate in terms of developing predictive models. In particular, EAT volume has been shown in many studies to be a prognostic marker for overall survival in various cardiac diseases, and it remains unclear whether the robust determination of EAT area as a surrogate for EAT volume or the potentially less robust direct determination of EAT volume is better suited for the development of predictive models [1,2,3,4,5,6,7]. It is therefore a future task to investigate the impact of 2D versus 3D measurements for outcome analysis.

Our study has several limitations. First, all investigated methods were developed on images from diastolic phases acquired in ECG-gated spiral technique. Future studies should address whether the presented algorithm is also applicable to coronary CT angiography (CCTA) data. In contrast to pre-interventional CT for planning of TAVR and surgery, CCTA data are typically acquired with step-and-shoot technique and reconstructed in various heart phases. Images acquired in step-and-shoot frequently show step artifacts, potentially affecting automated localization. This may require the inclusion of such data in the training set. Moreover, data was obtained from only one center. Although methods were developed based on various CT scans with different scan lengths (ECG-gated thoraco-abdominal and cardiac CT scans), a multi-center study is desirable to further verify the generalizability. Even though this study considered a heterogeneous dataset, it did not examine how scanning protocol differences and image quality affect the EAT measurement. In addition, only contrast-enhanced CT scans were included in our study, as these are part of the pre-interventional diagnostic work-up in patients undergoing TAVR or heart surgery.

In conclusion, we propose a pipeline for robust automatic 2D assessment of EAT in pre-interventional, ECG-gated spiral CT scans. End-to-end automation eliminates the need for any manual interaction and enables an opportunistic large-scale assessment of EAT.

Data availability
The trained models can only be shared for research purposes on request due to the German Data Protection Law.

Code availability
The code used for this study is publicly available at https://github.com/ukb-rad-cfqiai/EAT_Assessment, which allows for reproducing the described experiments with own data.

Abbreviations
AV:
Aortic valve

BOA:
Body and organ analysis

CAT:
Cardiac adipose tissue

CI:
Confidence intervals

CNN:
Convolutional neural network

CCTA:
Coronary computed tomography angiography

DL:
Deep learning

EAT:
Epicardial adipose tissue

ED:
Euclidean distance

HU:
Hounsfield units

MAD:
Mean absolute difference

PAT:
Pericardial adipose tissue

RL:
Reinforcement learning

TAVR:
Transcatheter aortic valve replacement

References
Mancio J, Azevedo D, Saraiva F et al (2018) Epicardial adipose tissue volume assessed by computed tomography and coronary artery disease: a systematic review and meta-analysis. Eur Heart J Cardiovasc Imaging 19:490–497

Article
 
PubMed
 
Google Scholar
 

Bastarrika G, Broncano J, Schoepf UJ et al (2010) Relationship between coronary artery disease and epicardial adipose tissue quantification at cardiac CT. Acad Radiol 17:727–734

Article
 
PubMed
 
Google Scholar
 

Eisenberg E, McElhinney PA, Commandeur F et al (2020) Deep learning–based quantification of epicardial adipose tissue volume and attenuation predicts major adverse cardiovascular events in asymptomatic subjects. Circ Cardiovasc Imaging 13:e009829

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Miller RJH, Shanbhag A, Killekar A et al (2024) AI-derived epicardial fat measurements improve cardiovascular risk prediction from myocardial perfusion imaging. NPJ Digit Med 7:1–8

Article
 
Google Scholar
 

Mahabadi AA, Berg MH, Lehmann N et al (2013) Association of epicardial fat with cardiovascular risk factors and incident myocardial infarction in the general population. J Am Coll Cardiol 61:1388–1395

Article
 
PubMed
 
Google Scholar
 

Eberhard M, Stocker D, Meyer M et al (2019) Epicardial adipose tissue volume is associated with adverse outcomes after transcatheter aortic valve replacement. Int J Cardiol 286:29–35

Article
 
PubMed
 
Google Scholar
 

Schulz A, Beuthner BE, Böttiger ZM et al (2024) Epicardial adipose tissue as an independent predictor of long-term outcome in patients with severe aortic stenosis undergoing transcatheter aortic valve replacement. Clin Res Cardiol. https://doi.org/10.1007/s00392-024-02387-5

Salam B, Al-Kassou B, Weinhold L et al (2024) CT-derived epicardial adipose tissue inflammation predicts outcome in patients undergoing transcatheter aortic valve replacement. J Thorac Imaging. https://doi.org/10.1097/RTI.0000000000000776

Fukushima T, Maetani T, Chubachi S et al (2024) Epicardial adipose tissue measured from analysis of adipose tissue area using chest CT imaging is the best potential predictor of COVID-19 severity. Metabolism 150:155715

Article
 
CAS
 
PubMed
 
Google Scholar
 

Nowak S, Faron A, Luetkens JA et al (2020) Fully automated segmentation of connective tissue compartments for CT-based body composition analysis: a deep learning approach. Invest Radiol 55:357

Article
 
CAS
 
PubMed
 
Google Scholar
 

Faron A, Opheys NS, Nowak S et al (2021) Deep learning-based body composition analysis predicts outcome in melanoma patients treated with immune checkpoint inhibitors. Diagnostics 11:2314

Article
 
CAS
 
PubMed
 
PubMed Central
 
Google Scholar
 

Magudia K, Bridge CP, Bay CP et al (2021) Population-scale CT-based body composition analysis of a large outpatient population using deep learning to derive age-, sex-, and race-specific reference curves. Radiology 298:319–329

Article
 
PubMed
 
Google Scholar
 

Nowak S, Theis M, Wichtmann BD et al (2022) End-to-end automated body composition analyses with integrated quality control for opportunistic assessment of sarcopenia in CT. Eur Radiol 32:3142–3151

Article
 
PubMed
 
Google Scholar
 

Salam B, Al Zaidi M, Sprinkart AM et al (2023) Opportunistic CT-derived analysis of fat and muscle tissue composition predicts mortality in patients with cardiogenic shock. Sci Rep 13:22293

Article
 
CAS
 
PubMed
 
PubMed Central
 
Google Scholar
 

Nowak S, Kloth C, Theis M et al (2024) Deep learning–based assessment of CT markers of sarcopenia and myosteatosis for outcome assessment in patients with advanced pancreatic cancer after high-intensity focused ultrasound treatment. Eur Radiol 34:279–286

Article
 
PubMed
 
Google Scholar
 

Commandeur F, Goeller M, Razipour A et al (2019) Fully automated CT quantification of epicardial adipose tissue by deep learning: a multicenter study. Radiol Artif Intell 1:e190045

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Li X, Sun Y, Xu L et al (2021) Automatic quantification of epicardial adipose tissue volume. Med Phys 48:4279–4290

Article
 
PubMed
 
Google Scholar
 

Hoori A, Hu T, Lee J, Al-Kindi S, Rajagopalan S, Wilson DL (2022) Deep learning segmentation and quantification method for assessing epicardial adipose tissue in CT calcium score scans. Sci Rep 12:2276

Article
 
CAS
 
PubMed
 
PubMed Central
 
Google Scholar
 

Haubold J, Baldini G, Parmar V et al (2023) BOA: a CT-based body and organ analysis for radiologists at the point of care. Invest Radiol. https://doi.org/10.1097/RLI.0000000000001040

Wasserthal J, Breit H-C, Meyer MT et al (2023) TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol Artif Intell 5:e230024

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH (2021) nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18:203–211

Article
 
CAS
 
PubMed
 
Google Scholar
 

Oyama N, Goto D, Ito YM et al (2011) Single-slice epicardial fat area measurement: do we need to measure the total epicardial fat volume? Jpn J Radiol 29:104–109

Article
 
PubMed
 
Google Scholar
 

Vach M, Luetkens JA, Faron A et al (2023) Association between single-slice and whole heart measurements of epicardial and pericardial fat in cardiac MRI. Acta Radiol 64:2229–2237

Article
 
PubMed
 
Google Scholar
 

Bridge CP, Rosenthal M, Wright B et al (2018) Fully-automated analysis of body composition from CT in cancer patients using convolutional neural networks. In: Stoyanov D, Taylor Z, Sarikaya D et al (eds) Proceedings of OR 2.0 context-aware operating theaters, computer assisted robotic endoscopy, clinical image-based procedures, and skin image analysis. pp 204–213

Ghesu FC, Georgescu B, Mansi T, Neumann D, Hornegger J, Comaniciu D (2016) An artificial agent for anatomical landmark detection in medical images. In: Ourselin S, Joskowicz L, Sabuncu MR et al (eds) Proceedings of medical image computing and computer-assisted intervention (MICCAI). pp 229–237

Leroy G, Rueckert D, Alansary A (2020) Communicative reinforcement learning agents for landmark detection in brain images. In: Kia SM, Mohy-ud-Din H, Abdulkadir A et al (eds) Proceedings of machine learning in clinical neuroimaging and radiogenomics in neuro-oncology (MLCN). pp 177–186

Kang SH, Jeon K, Kang S-H, Lee S-H (2021) 3D cephalometric landmark detection by multiple stage deep reinforcement learning. Sci Rep 11:17509

Article
 
CAS
 
PubMed
 
PubMed Central
 
Google Scholar
 

Hollstein R (2023) Reinforcement learning. In: Optimierungsmethoden: einführung in die klassischen, naturanalogen und neuronalen optimierungen. Springer Vieweg, pp 331–350

Fedorov A, Beichel R, Kalpathy-Cramer J et al (2012) 3D Slicer as an image computing platform for the quantitative imaging network. Magn Reson Imaging 30:1323–1341

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Tan M, Le Q (2019) EfficientNet: rethinking model scaling for convolutional neural networks. PMLR 2019:6105–6114

Alansary A, Oktay O, Li Y et al (2019) Evaluating reinforcement learning agents for anatomical landmark detection. Med Image Anal 53:156–164

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Leys C, Ley C, Klein O, Philippe B, Laurent L (2013) Detecting outliers: do not use standard deviation around the mean, use absolute deviation around the median. J Exp Soc Psychol 49:764–766

Article
 
Google Scholar
 

Cardoso MJ, Li W, Brown R et al (2022) MONAI: an open-source framework for deep learning in healthcare. Preprint at https://doi.org/10.48550/arXiv.2211.02701

Kerfoot E, Clough J, Oksuz I, Lee J, King AP, Schnabel JA (2019) Left-ventricle quantification using residual U-Net. In: Pop M, Sermesant M, Zhao J et al (eds) Proceedings of statistical atlases and computational models of the heart. Atrial segmentation and LV quantification challenges—STACOM 2018. pp 371–380

Rodrigues ÉO, Morais FFC, Morais NAOS, Conci LS, Neto LV, Conci A (2016) A novel approach for the automated segmentation and volume quantification of cardiac fats on computed tomography. Comput Methods Programs Biomed 123:109–128

Article
 
CAS
 
PubMed
 
Google Scholar
 

Benčević M, Galić I, Habijan M, Pižurica A (2022) Recent progress in epicardial and pericardial adipose tissue segmentation and quantification based on deep learning: a systematic review. Appl Sci 12:5217

Article
 
Google Scholar
 

Shahzad R, Bos D, Metz C et al (2013) Automatic quantification of epicardial fat volume on non‐enhanced cardiac CT scans using a multi‐atlas segmentation approach. Med Phys 40:091910

Article
 
PubMed
 
Google Scholar
 

Chan J, Thakur U, Tan S et al (2023) Inter-software and inter-scan variability in measurement of epicardial adipose tissue: a three-way comparison of a research-specific, a freeware and a coronary application software platform. Eur Radiol 33:8445–8453

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Zhang Q, Zhou J, Zhang B, Jia W, Wu E (2020) Automatic epicardial fat segmentation and quantification of CT scans using dual U-Nets with a morphological processing layer. IEEE Access 8:128032–128041

Article
 
Google Scholar
 

West HW, Siddique M, Williams MC et al (2023) Deep-learning for epicardial adipose tissue assessment with computed tomography. JACC Cardiovasc Imaging 16:800–816

Article
 
PubMed
 
PubMed Central
 
Google Scholar
 

Santos Da Silva G, Casanova D, Oliva JT, Rodrigues EO (2024) Cardiac fat segmentation using computed tomography and an image-to-image conditional generative adversarial neural network. Med Eng Phys 124:104104

Article
 
PubMed
 
Google Scholar
 

Greif M, Becker A, von Ziegler F et al (2009) Pericardial adipose tissue determined by dual source CT is a risk factor for coronary atherosclerosis. Arterioscler Thromb Vasc Biol 29:781–786

Article
 
CAS
 
PubMed
 
Google Scholar
 

Download references

Funding
M.T. was funded by RACOON (NUM 2.0), which is supported by the Federal Ministry of Education and Research of Germany (grant no. 01KX2121). The funders had no influence on the conception and design of the study, data analysis, data collection, preparation of the manuscript, and decision to publish. Open Access funding enabled and organized by Projekt DEAL.

Author information
Author notes
Maike Theis and Laura Garajová contributed equally to this work.

Authors and Affiliations
Department of Diagnostic and Interventional Radiology, University Hospital Bonn, Bonn, Germany

Maike Theis, Laura Garajová, Babak Salam, Sebastian Nowak, Wolfgang Block, Ulrike I. Attenberger, Daniel Kütting, Julian A. Luetkens & Alois M. Sprinkart

Department of Radiotherapy and Radiation Oncology, University Hospital Bonn, Bonn, Germany

Wolfgang Block

Department of Neuroradiology, University Hospital Bonn, Bonn, Germany

Wolfgang Block

Contributions
M.T. and L.G. were responsible for the development of methods and were the major contributors to writing the manuscript. S.N. and A.M.S. supported and advised method development and evaluation. B.S. was responsible for data annotation supervised by D.K. W.B. assisted with data curation. J.A.L., with his medical expertise in cardiac imaging and transcatheter aortic valve replacement, supported the method development and writing of the manuscript. The entire study was supervised and administrated by A.M.S., who also mainly revised the first draft of the manuscript. U.I.A. supported project administration and was responsible for funding acquisition. All authors read and approved of the final manuscript.

Corresponding author
Correspondence to Maike Theis.

Ethics declarations
Ethics approval and consent to participate
This retrospective single-center study was approved by the local Ethics Committees at the Medical Faculty of the Rheinische Friedrich-Wilhelms-Universität Bonn with a waiver for written informed consent.

Consent for publication
Not applicable.

Competing interests
The authors declare that they have no competing interests.

Additional information
Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Supplementary information
ELECTRONIC SUPPLEMENTARY MATERIAL
Rights and permissions
Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

Reprints and permissions

About this article
Check for updates. Verify currency and authenticity via CrossMark
Cite this article
Theis, M., Garajová, L., Salam, B. et al. Deep learning for opportunistic, end-to-end automated assessment of epicardial adipose tissue in pre-interventional, ECG-gated spiral computed tomography. Insights Imaging 15, 301 (2024). https://doi.org/10.1186/s13244-024-01875-6

Download citation

Received
12 July 2024

Accepted
30 November 2024

Published
19 December 2024

DOI
https://doi.org/10.1186/s13244-024-01875-6

Share this article
Anyone you share the following link with will be able to read this content:

Get shareable link
Provided by the Springer Nature SharedIt content-sharing initiative

Keywords
Epicardial adipose tissue
Transcatheter aortic valve replacement
Tomography (X-ray computed)
Deep learning
Download PDF
Sections
Figures
References
Abstract
Introduction
Materials and methods
Results
Discussion
Data availability
Code availability
Abbreviations
References
Funding
Author information
Ethics declarations
Additional information
Supplementary information
Rights and permissions
About this article
Advertisement

Support and Contact
Jobs
Language editing for authors
Scientific editing for authors
Leave feedback
Terms and conditions
Privacy statement
Accessibility
Cookies
Follow SpringerOpen
SpringerOpen Twitter page
SpringerOpen Facebook page
By using this website, you agree to our Terms and Conditions, Your US state privacy rights, Privacy statement and Cookies policy. Your privacy choices/Manage cookies we use in the preference centre.

Springer Nature
© 2025 BioMed Central Ltd unless otherwise stated. Part of Springer Nature.