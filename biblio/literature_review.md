# Literature Review: Multi-Dimensional and Multi-Temporal Deforestation Risk Prediction in the Congo Basin

**Last updated:** 2026-02-27

---

## 1. Introduction

Tropical deforestation remains one of the most pressing environmental challenges of the 21st century, contributing approximately 10% of global CO₂ emissions and driving biodiversity loss at unprecedented rates. The Congo Basin, Earth's second-largest tropical forest after the Amazon, spans six countries and stores an estimated 60 billion tonnes of carbon. Yet, compared to the Amazon and Southeast Asia, it remains critically understudied in the predictive modeling literature.

This review synthesizes the state of knowledge across seven dimensions relevant to our research project: (1) deforestation patterns in the Congo Basin, (2) predictive modeling approaches, (3) multi-dimensional driver analysis, (4) machine learning interpretability, (5) spatial-temporal validation, (6) remote sensing data sources, and (7) emerging approaches including foundation models.

---

## 2. Deforestation in the Congo Basin

### 2.1 Extent and Trends

The Hansen Global Forest Change dataset (Hansen et al., 2013) provides the foundational reference for monitoring forest loss at 30 m resolution globally. Using Landsat time-series imagery from 2000 to the present (now v1.12), it maps annual tree cover loss across the tropics. This dataset underpins nearly all recent quantitative work on deforestation.

In the Congo Basin specifically, Tyukavina et al. (2018) demonstrated that **84% of forest disturbance is due to small-scale, nonmechanized clearing for agriculture**, with smallholder clearing in the DRC alone accounting for nearly two-thirds of total basin-wide forest loss. Critically, annual rates of small-scale clearing in primary forests doubled between 2000 and 2014, mirroring population growth trajectories.

Potapov et al. (2012) quantified forest cover loss in the DRC using Landsat ETM+ data (2000–2010), establishing baseline deforestation rates for the country. Ernst et al. (2013) provided a multi-temporal analysis of national forest cover change across the basin for 1990, 2000, and 2005, documenting both deforestation and regeneration dynamics. Aleman et al. (2018) placed these trends in historical context, analyzing forest extent changes in tropical Africa since 1900.

More recently, Kafy et al. (2024) projected forest cover changes in the Congo Basin under IPCC climate scenarios through 2050, estimating a 3.7–4.0% loss in dense forest cover under all scenarios.

### 2.2 Drivers: Smallholder Agriculture, Mining, Logging, Roads

Shapiro et al. (2023) confirmed that small-scale agriculture continues to drive deforestation even in fragmented forest landscapes (2015–2020). Verhegghen et al. (2012) provided essential vegetation type mapping from multi-sensor time series, establishing a spatial framework for understanding forest composition across the basin.

Road development is an increasingly recognized indirect driver. Masolele et al. (2024) used Sentinel-1 and Sentinel-2 imagery with deep learning to monitor road development at 10 m monthly resolution across the Congo Basin, demonstrating the link between infrastructure expansion and subsequent forest clearing.

Mining represents another growing threat: deforestation impacts around mining sites are on average **28 times larger than the mine footprint itself**, as settlements and farmland spring up around extraction operations.

### 2.3 Socioeconomic and Governance Context

Population growth, agricultural commodity demand, weak governance, and conflict interact to drive deforestation in the Congo Basin. The DRC's low development and political instability produce clearing rates largely correlated with population growth (Tyukavina et al., 2018). Poor governance and corruption have been identified as the biggest obstacles to forest protection, enabling both legal and illegal industrial operations.

---

## 3. Predictive Modeling for Deforestation

### 3.1 Traditional Approaches

Early deforestation models relied on logistic regression, spatial econometrics, and Markov chain models to predict land-use transitions. These approaches, while interpretable, struggle to capture complex nonlinear interactions among drivers and often assume stationarity in driver–response relationships.

### 3.2 Machine Learning: Random Forest and XGBoost

Gradient boosted trees, particularly XGBoost (Chen & Guestrin, 2016), have emerged as a dominant approach for tabular geospatial prediction tasks. XGBoost offers several advantages: handling of missing values, regularization against overfitting, and native support for feature importance. Mayfield et al. (2017) demonstrated the effectiveness of freely available datasets combined with ML methods for deforestation prediction, establishing a practical framework for countries with limited resources.

Saha et al. (2024) applied machine learning to predict Amazon deforestation dynamics, showing the power of integrating socioeconomic and environmental variables in a unified predictive framework.

### 3.3 Deep Learning: CNNs, ConvLSTM, Vision Transformers

**DeepForestcast** (Ball et al., 2022) represents a landmark study in deforestation forecasting. Using deep CNNs trained on Hansen maps (2001–2020), Landsat imagery, and the ALOS digital surface model, it forecast future deforestation in the Amazon at ~30 m resolution. Four architectures were tested (2D-CNN, 3D-CNN, ConvLSTM, hybrid), achieving **F1 scores of 0.58–0.71**. Critically, models that retained spatiotemporal structure outperformed those using only spatial patterns.

**ForestCast** (Reiersen et al., 2025), developed by Google DeepMind, scales this approach using vision transformers. Operating on Landsat/Sentinel-2 tiles with change history, it processes entire tiles in one pass, matching or exceeding accuracy of methods requiring specialized inputs (e.g., road networks). The ForestCast Southeast Asia dataset is the first public benchmark for deep learning deforestation risk forecasting.

Irvin et al. (2024) tackled the complementary problem of deforestation **driver attribution** using deep learning on satellite imagery, automating a task previously requiring labor-intensive manual labeling. Descals et al. (2024) released a labeled dataset for direct deforestation driver classification in Cameroon.

### 3.4 Systematic Reviews and Benchmarks

Mugabowindekwe et al. (2024) conducted a systematic review of deep learning for deforestation detection, cataloging architectures (U-Net, DeepLab V3, ResNet, SegNet) and their performance across biomes. Ortiz-Reyes et al. (2021) compared spatial and temporal deep learning methods for land-use classification following deforestation across the pan-tropics.

---

## 4. Multi-Dimensional Drivers and Feature Engineering

### 4.1 Satellite Imagery and Vegetation Indices

Sentinel-2 (Drusch et al., 2012) provides high-resolution (10 m) multispectral imagery with 13 bands and ~5-day revisit, making it ideal for near-real-time monitoring. The Hansen dataset uses Landsat time series at 30 m resolution for annual change detection.

### 4.2 Climate Variables

**CHIRPS** (Funk et al., 2015) provides quasi-global precipitation at 5 km daily resolution from 1981 to near-present, combining infrared cold cloud duration observations with station data. **ERA5** (Hersbach et al., 2020) and **ERA5-Land** (Muñoz-Sabater et al., 2021) provide comprehensive atmospheric reanalysis at 9 km resolution, including temperature, soil moisture, and evapotranspiration — variables linked to both agricultural suitability and fire risk.

### 4.3 Socioeconomic Indicators

**WorldPop** (Tatem, 2017) disaggregates census population counts to ~100 m grid cells using machine learning, providing annually updated population density estimates. **VIIRS nighttime lights** (Elvidge et al., 2017) serve as a proxy for economic activity, electrification, and urbanization at 750 m resolution.

Commodity prices (World Bank Pink Sheet) and exchange rates provide macro-economic context, while national-level indicators (GDP, inflation) capture broader economic dynamics.

### 4.4 Conflict and Governance

**ACLED** (Raleigh et al., 2010) provides georeferenced data on armed conflict events, including battles, violence against civilians, and riots. Armed conflict interacts with deforestation through displacement, breakdown of governance, and opportunistic resource extraction.

**V-Dem** (Coppedge et al., 2024) offers 531 indicators of democracy and governance, while the World Governance Indicators provide measures of corruption, rule of law, and political stability. Busch & Ferretti-Gallon (2017) found in their meta-analysis that **policies directly regulating land use** (protected areas, forest law enforcement) consistently reduce deforestation, whereas general governance indicators show inconsistent effects.

### 4.5 Integration of Heterogeneous Data Sources

Curtis et al. (2018) classified drivers of global forest loss into four categories (commodity agriculture 27%, forestry 26%, shifting agriculture 24%, wildfire 23%), establishing a typology for driver attribution. Armenteras et al. (2019) demonstrated that deforestation drivers are highly **scale- and context-dependent**, with the same driver producing different effects in different geographies — motivating multi-scale analysis approaches.

Pendrill et al. (2019) quantified the role of international trade in driving tropical deforestation, finding that 29–39% of deforestation emissions are embodied in traded commodities. Hosonuma et al. (2012) provided a comprehensive assessment of proximate deforestation and degradation drivers across developing countries, distinguishing commercial agriculture, subsistence agriculture, and extractive activities.

---

## 5. Model Interpretability

### 5.1 SHAP Values and Feature Importance

SHAP (SHapley Additive exPlanations) (Lundberg & Lee, 2017) provides a unified framework for interpreting predictions based on cooperative game theory. Unlike permutation importance, SHAP values are locally consistent, additive, and provide both global and local explanations. Lundberg et al. (2020) extended this to tree-based models specifically, enabling efficient computation for XGBoost.

### 5.2 Application to Environmental Science

The XGBoost + SHAP combination has become a standard approach in environmental modeling. It enables identification of nonlinear relationships and threshold effects between drivers and outcomes. For deforestation, SHAP can reveal which drivers dominate at different spatial scales and how their importance varies temporally — directly supporting the ablation study design in our project.

### 5.3 Ablation Studies for Driver Attribution

Systematic ablation — removing feature groups and measuring performance changes — quantifies the **marginal contribution of each dimension** (satellite, climate, socioeconomic, conflict, governance). This approach, combined with SHAP, provides both statistical and causal insight into driver dynamics. Molnar (2022) provides a comprehensive framework for interpretable ML methods including ablation.

---

## 6. Spatial and Temporal Validation

### 6.1 Spatial Autocorrelation and Data Leakage

A critical methodological concern in geospatial ML is spatial autocorrelation. **Ploton et al. (2020)** demonstrated that random cross-validation can produce dramatically inflated performance estimates for ecological models: a random forest predicting above-ground biomass appeared highly accurate with random splits but showed **no predictive skill** under spatial cross-validation. This finding has profound implications for deforestation prediction models.

### 6.2 Blocked Cross-Validation Strategies

Roberts et al. (2017) established the theoretical and practical framework for block cross-validation with spatial data, recommending that **block CV should be used whenever dependence structures exist**, even if no correlation is visible in residuals. Four blocking strategies are discussed: spatial, hierarchical, phylogenetic, and predictor-space blocking.

### 6.3 Temporal Validation and Forecasting

For deforestation prediction specifically, temporal splits are essential: training on historical data and testing on future years avoids temporal information leakage. Our project's temporal split (train ≤2020, validation 2021, test 2022) follows this best practice. The debate continues: Wadoux et al. (2021) argued that spatial cross-validation may **overcorrect** in some contexts, while Meyer & Pebesma (2021) introduced the concept of "area of applicability" to quantify where models can be trusted to extrapolate.

Meyer et al. (2024) provide a comprehensive review of challenges in data-driven geospatial modeling, synthesizing current best practices for environmental research.

---

## 7. Emerging Approaches

### 7.1 Foundation Models for Earth Observation

**Prithvi-EO-2.0** (Jakubik et al., 2024), developed by NASA and IBM, is a 600M-parameter multi-temporal foundation model pre-trained on 4.2M global time series samples from Harmonized Landsat-Sentinel-2 data. It incorporates temporal and location embeddings and outperforms its predecessor by 8% across diverse tasks. Mai et al. (2023) discuss broader opportunities and challenges for foundation models in geospatial AI.

### 7.2 Benchmarking and Evaluation

GEO-Bench (Lacoste et al., 2024) establishes a standardized benchmark for evaluating geospatial foundation models, enabling fair comparison across architectures and pre-training strategies.

### 7.3 Scalable Risk Mapping

ForestCast (Reiersen et al., 2025) represents the current state of the art in scalable deforestation risk mapping, using vision transformers to process entire satellite tiles and produce spatially explicit probability maps. Its "pure satellite" approach, requiring only Landsat/Sentinel-2 inputs and Hansen change history, demonstrates that sophisticated features (roads, population density) may be implicitly captured by satellite imagery.

---

## 8. Gaps and Research Opportunities

This review identifies several key gaps that our project aims to address:

1. **Geographic bias**: The vast majority of deforestation prediction studies focus on the Amazon or Southeast Asia. The Congo Basin — with its distinct driver profile (smallholder-dominated, conflict-affected, rapidly growing population) — is **critically underrepresented** in the ML prediction literature.

2. **Limited feature dimensionality**: Most models use satellite imagery alone or combine it with a few auxiliary layers (elevation, distance to roads). Very few studies integrate the **full spectrum of drivers** simultaneously: satellite, climate, socioeconomic, demographic, conflict, governance, and cultural variables.

3. **Temporal scale analysis**: No existing study systematically examines **which temporal scales matter** for which drivers. Does an economic shock propagate to deforestation in 6 months or 3 years? Our triple ablation design (dimension × temporal scale × spatial resolution) is novel.

4. **Rigorous validation**: Many published models lack proper spatial cross-validation (Ploton et al., 2020). Combining spatial blocking with temporal out-of-sample testing remains rare.

5. **Interpretability gap**: Deep learning models (DeepForestcast, ForestCast) achieve strong performance but offer limited insight into driver dynamics. XGBoost + SHAP provides a complementary interpretable approach that can inform policy.

---

## Reference Summary Table

| Key | Year | Theme | Method | Region | Relevance |
|-----|------|-------|--------|--------|-----------|
| Hansen2013 | 2013 | Forest monitoring | Landsat time series | Global | Primary deforestation dataset (GFC v1.12) |
| Tyukavina2018 | 2018 | Congo deforestation | Remote sensing | Congo Basin | 84% loss = smallholder clearing |
| Shapiro2023 | 2023 | Congo drivers | Remote sensing | Congo Basin | Fragmented forest dynamics 2015–2020 |
| Ernst2013 | 2013 | Forest change | Multi-temporal mapping | Congo Basin | National-level change 1990–2005 |
| Potapov2012 | 2012 | DRC deforestation | Landsat ETM+ | DRC | Baseline forest loss 2000–2010 |
| Aleman2018 | 2018 | Historical forests | Meta-analysis | Tropical Africa | Century-scale forest extent |
| Kafy2024 | 2024 | Future projections | Random Forest + GEE | Congo Basin | 2050 scenarios IPCC |
| Ball2022 | 2022 | Prediction | Deep CNN / ConvLSTM | Amazon | DeepForestcast, F1=0.58–0.71 |
| ForestCast2025 | 2025 | Prediction | Vision Transformer | SE Asia | Google, scalable risk mapping |
| Irvin2024 | 2024 | Driver attribution | Deep learning | Pantropical | Automatic driver classification |
| Descals2024 | 2024 | Driver dataset | Labeled imagery | Cameroon | Training data for Congo Basin |
| Masolele2024 | 2024 | Road monitoring | DL + Sentinel-1/2 | Congo Basin | Roads as indirect driver |
| Mugabowindekwe2024 | 2024 | Review | Systematic review | Global | DL for deforestation detection |
| Saha2024 | 2024 | Prediction | ML ensemble | Amazon | Multi-variable prediction |
| Mayfield2017 | 2017 | Prediction | ML methods | Global | Freely available data + ML |
| Curtis2018 | 2018 | Driver typology | Remote sensing | Global | 4-class driver classification |
| Busch2017 | 2017 | Driver meta-analysis | Econometric review | Tropics | What drives/stops deforestation |
| Armenteras2019 | 2019 | Scale dependency | Spatial econometrics | Tropics | Drivers are scale-dependent |
| Pendrill2019 | 2019 | Trade & deforestation | Trade modeling | Tropics | 29–39% emissions from trade |
| Hosonuma2012 | 2012 | Driver assessment | REDD+ synthesis | Developing countries | Proximate driver typology |
| Ordway2017 | 2017 | Commodities | Remote sensing | Sub-Saharan Africa | Crop expansion risk |
| Chen2016 | 2016 | ML method | Gradient boosting | — | XGBoost algorithm |
| Lundberg2017 | 2017 | Interpretability | Game theory | — | SHAP framework |
| Lundberg2020 | 2020 | Interpretability | TreeSHAP | — | Efficient SHAP for trees |
| Molnar2022 | 2022 | Interpretability | Review/Book | — | IML comprehensive guide |
| Friedman2001 | 2001 | ML method | Gradient boosting | — | Foundational gradient boosting |
| Roberts2017 | 2017 | Validation | Block CV | — | Spatial CV framework |
| Ploton2020 | 2020 | Validation | Spatial validation | Tropics | Inflated accuracy from random CV |
| Wadoux2021 | 2021 | Validation | Debate | — | Critique of spatial CV |
| Meyer2021 | 2021 | Validation | Area of applicability | — | Extrapolation boundaries |
| Meyer2024 | 2024 | Challenges | Review | Global | Geospatial modeling best practices |
| Gorelick2017 | 2017 | Platform | Cloud computing | Global | Google Earth Engine |
| Funk2015 | 2015 | Climate data | Precipitation | Global | CHIRPS dataset |
| Hersbach2020 | 2020 | Climate data | Reanalysis | Global | ERA5 reanalysis |
| MunozSabater2021 | 2021 | Climate data | Land reanalysis | Global | ERA5-Land |
| Drusch2012 | 2012 | Satellite data | Optical mission | Global | Sentinel-2 mission |
| Farr2007 | 2007 | Elevation data | Radar | Global | SRTM DEM |
| Tatem2017 | 2017 | Population data | ML disaggregation | Global | WorldPop 100m |
| Elvidge2017 | 2017 | Socioeconomic proxy | Nighttime lights | Global | VIIRS DNB |
| Raleigh2010 | 2010 | Conflict data | Event coding | Global | ACLED dataset |
| VDem2024 | 2024 | Governance data | Expert survey | Global | V-Dem 531 indicators |
| Giglio2016 | 2016 | Fire data | Active fire detection | Global | MODIS Collection 6 |
| Jakubik2024 | 2024 | Foundation model | ViT + temporal embed | Global | Prithvi-EO-2.0 |
| Mai2023 | 2023 | Foundation models | Review | — | Opportunities & challenges |
| Lacoste2024 | 2024 | Benchmark | Evaluation framework | Global | GEO-Bench |

---

*This literature review accompanies `references.bib` for use in future publications.*
