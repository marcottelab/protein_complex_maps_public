protein_complex_maps
====================

#Scripts for handling protein complex map data

##Elution correlation
###Correlation matrices
for each experiment, each species, and all experiments concatenated

`python ./protein_complex_maps/external/infer_complexes/score.py`

**input**:
      tab separated wide elution profile: prot_ids [tab] total_spectral_count [tab] frac1_spectral_count [tab] ...

**output**:
      corr_poisson

*output is a giant all by all matrix*

**Example**
```
python ./protein_complex_maps/external/infer_complexes/score.py ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt poisson
```

###Reformat all by all to tidy (3 column)

`python ./protein_complex_maps/features/convert_correlation.py`

**input**:
      corr_poisson

**output**:
      corr_poisson.pairs

*P1 P2 correlation_coefficient; For all protein pairs*

**Example**
```
python ./protein_complex_maps/features/convert_correlation.py --input_correlation_matrix ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt.corr_poisson --input_elution_profile ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt --output_file ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt.corr_poisson_tidy
```

###Feature matrix

*Any feature which you can put on a pair of proteins*

`python ./protein_complex_maps/features/build_feature_matrix.py`

**input**:
       all .corr_poisson.pairs

**output**:
      feature_matrix.txt

*Note: this is the point to put in additional features like AP-MS etc. as long as it describes a pair of proteins*

|pairs |Feature1 |Feature2 |Feature3|
|--- | --- | --- | --- |
|P1 P2 |value1 |value2 |value3|
|...|...|...|...|
|PN PN-1|value4|value5|value6|

*n x m, where n = #prots choose 2, m = # of features*

**Example**
```
python ./protein_complex_maps/features/build_feature_matrix.py --input_pairs_files ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt.corr_poisson_tidy --output_file ./examples/Hs_helaN_ph_hcw120_2_psome_exosc_randos.txt.corr_poisson_tidy.featmat
```

###Format corum into test and training sets
*Remove redundancy from corum (merge similar clusters)*

`python ./protein_complex_maps/complex_merge.py`

**input**:
      nonredundant_allComplexesCore_mammals.txt
**output**:
     nonredundant_allComplexesCore_mammals_merged06.txt

*Randomly split the corum complexes into training and test (split)*

`python ./protein_complex_maps/features/split_complexes.py`

**input**: complexes nonredundant_allComplexesCore_mammals_merged06.txt

**output**:
+ [input_basename].test.txt
+ [input_basename].train.txt
+ [input_basename].test_ppis.txt
+ [input_basename].train_ppis.txt
+ [input_basename].neg_test_ppis.txt
+ [input_basename].neg_train_ppis.txt

*Takes any pairwise overlap between train and test ppi, and randomly removes ppi from either test or train. 
So say complex 1 = AB, AC, BC & complex 2 = AB AC AD BC BD => complex 1 = AB BC, complex 2 = AB AD CD
Also make sure complexes between training and test are completely separated*

**Example**
```
python ./protein_complex_maps/complex_merge.py --cluster_filename ./examples/allComplexesCore_geneid.txt --output_filename ./examples/allComplexesCore_geneid_merged06.txt --merge_threshold 0.6
python ./protein_complex_maps/features/split_complexes.py --input_complexes ./examples/allComplexesCore_geneid_merged06.txt
```


###Make feature matrix w/ labels from corum 

`python ./protein_complex_maps/features/add_label.py`

**input**:
     feature_matrix.txt

**output**:
      corum_train_labeled.txt

*(These are the possible labels)*
+ *+1 positive label = pair is co-complex in corum*
+ *-1 negative label = pair is in corum, but not in same complex*
+ *0 = at least one protein in the pair is not in corum*

###Make input for the SVM

*Convert to libsvm format training set, strips out a lot of headers, etc.*

`python ./protein_complex_maps/features/feature2libsvm.py`

**input**:
      corum_train_labeled.txt

**output**:
      corum_train_labeled.libsvm1.txt, tab separated


*SVM biased toward large numbers in features. Scaling just puts all features scaled to 1.*

`$LIBSVM_HOME/svm-scale`

**input**:
      corum_train_labeled.libsvm1.scale_parameters

**output**:
      corum_train_labeled.libsvm1.scale.txt


*SVM training and parameter sweep to optimize C and gamma*

*parameter sweep using training set (trains on 9/10th, compared to leave out)*

`python $LIBSVM_HOME/tools/grid.py`

**input**:
      corum_train_labeled.libsvm1.scale.txt

**output**:
      corum_train_labeled.libsvm1.scale.txt.out


###Train classifier

*Takes optimal c and g from SVM training and trains a classifier*

`$LIBSVM_HOME/svm-train`

**input**:
      corum_train_labeled.libsvm1.scale.txt

**output**:
      corum_train_labeled.libsvm1.scale.model_c_g (with c and g values)

*predict unlabeled set w/ test set on train model*

`$LIBSVM_HOME/svm-predict`

**input**:
      corum_train_labeled.libsvm0.scaleByTrain.txt, corum_train_labeled.libsvm1.scale.model_c_g

**output**:
      corum_train_labeled.libsvm0.scaleByTrain.resultsWprob

*probability ordered list of pairs*

`python ./protein_complex_maps/features/svm_results2pairs.py`

**inputs**:
      corum_train_labeled.txt, corum_train_labeled.libsvm0.scaleByTrain.resultsWprob

**output**:
      corum_train_labeled.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.txt


###Cluster PPis
*At this point, want to find clusters (dense regions)*

*two-stage clustering*

`python ./protein_complex_maps/features/clustering_parameter_optimization.py`

**inputs**:
+ corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.txt
     
+ nonredundant_allComplexesCore_mammals_merged06.train.txt

**outputs**:
+ corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.txt
+ corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.out

*Do a parameter sweep (about 1000 different possibilities*

+ PPi score threshold [1.0, 0.9., 0.8 ... .1]
+ Clusterone parameters 
    + overlap (jaccard score) [0.8, 0.7, 0.6]  -- merging complexes with overlap
    + density (threshold of total number of interactions vs. total possible interactions) unconnected -> fully connected
+ MCL inflation [1.2, 3, 4, 7]

**Process**: Run through clusterone, then run clusters from clusterone through MCL.

**Output**: a set of clusters times # of possible combinations

Select best set of clusters (usually a couple thousand) by comparing to corum training complex set using K-Cliques metric or other comparison metric

###Generate Cytoscape Network

*Make clusters into pairs*

`python ./protein_complex_maps/util/cluster2pairwise.py`

**input**:
corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].txt

**output**:
corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].pairsWclustID.txt

*Make clusters into node table*

`python ./protein_complex_maps/util/cluster2node_table.py`

**input**:
corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].txt
       
**output**:
corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].nodeTable.txt

*Make edge attribute table*

`python ./protein_complex_maps/util/pairwise2clusterid.py`

**inputs**:
+ corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.txt       
+ corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].txt

**output**:
corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.best_cluster_wOverlap_nr_allComplexesCore_mammals_psweep_clusterone_mcl.[best].edgeAttributeWClusterid.txt


*Load into Cytoscape*

