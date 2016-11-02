protein_complex_maps
====================

#Scripts for handling protein complex map data

##Elution correlation
###Correlation matrices
for each experiment, each species, and all experiments concatenated

`python ./protein_complex_maps/external/score.py`

**input**:
      tab separated wide elution profile

**output**:
      corr_poisson

*output is a giant all by all matrix*

###Reformat all by all to tidy (3 column)

`python ./protein_complex_maps/features/convert_correlation.py`

**input**:
      corr_poisson

**output**:
      corr_poisson.pairs

*P1 P2 correlation_coefficient; For all protein pairs*

###Feature matrix

(Any feature which you can put on a pair of proteins)
python ./protein_complex_maps/features/build_feature_matrix.py
input:
       all .corr_poisson.pairs
output:
      feature_matrix.txt

Note: this is the point to put in additional features like AP-MS etc. as long as it describes a pair of proteins
pairs Feature1 Feature2 Feature3
P1P3 Value2 value2 value3
...
n

n x m, where n = #prots choose 2, m = # of features

concatenation of individual .corr_poisson.pairs

###Format corum
Plant corum core mammal svm training
Randomly split the corum complexes into training and test (split)
Remove redundancy from corum (merge similar clusters)
(claire: not sure exactly how this was done)

 remove header from nonredundant_allComplexesCore_mammals_euNOG.csv

reduce redundancy in eunog corum complexes
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/complex_merge.py
input:
      nonredundant_allComplexesCore_mammals_euNOG.csv
output:
     nonredundant_allComplexesCore_mammals_euNOG_merged06.txt

/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/split_complexes.py

input: complexes nonredundant_allComplexesCore_mammals_euNOG_merged06.txt

output:

   Takes any pairwise overlap between train and test ppi, and randomly throw out from one. Train and test need to be totally different
   So say complex 1 = AB, AC, BC & complex 2 = AB AC AD BC BD => complex 1 = AB BC, complex 2 = AB AD CD
   Also make sure complexes between training and test are completely separated

Make feature matrix w/ labels from corum (euNOG corum) (/corum) current control script: arathtraesorysj_euNOG_run.sh
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/add_label.py
input:
     feature_matrix.txt
output:
      corum_train_labeled.txt

/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/add_label.py
input:
      feature_matrix.txt
output:
      corum_test_labeled.txt  #Is this ever used?

(These are the possible labels)
+1 positive label = pair is co-complex in corum
-1 negative label = pair is in corum, but not in same complex
0 = at least one protein in the pair is not in corum

Make input for the SVM
Convert to libsvm format training set, strips out a lot of headers, etc.
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/feature2libsvm.py
input:
      corum_train_labeled.txt
output:
      corum_train_labeled.libsvm1.txt, tab separated
pairs Feature1 Feature2 Feature3

Convert to libsvm format test set, strips out a lot of headers, etc.
 /home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/feature2libsvm.py
input:
      corum_test_labeled.txt
output:
      corum_test_labeled.libsvm1.txt, tab separated
-- keep_labels (0, 1, -1)

SVM biased toward large numbers in features. Scaling just puts all features scaled to 1.
/home/kdrew/programs/libsvm-3.20/svm-scale
input:
      corum_train_labeled.libsvm1.scale_parameters
output:
      corum_train_labeled.libsvm1.scale.txt
/home/kdrew/programs/libsvm-3.20/svm-scale
input:
      corum_test_labeled.libsvm1.scale_parameters
output:
      corum_test_labeled.libsvm1.scale.txt

SVM training and parameter sweep
(takes a long time)
Trained on training PPI set. Anything is Test PPis given 0's
##: parameter sweep using training set (trains on 9/10th, compared to leave out)
python ~/programs/libsvm-3.20/tools/grid.py
input:
      corum_train_labeled.libsvm1.scale.txt
output:
      corum_train_labeled.libsvm1.scale.txt.out

Train classifier
then make prediction on all the 0's (the unlabled) (currently run with train.sh)
Takes optimal c and g from SVM training and trains a classifier
/home/kdrew/programs/libsvm-3.20/svm-train
input:
      corum_train_labeled.libsvm1.scale.txt
output:
      corum_train_labeled.libsvm1.scale.model_c32_g05 (with c and g values)

predict unlabeled set w/ test set on train model
/home/kdrew/programs/libsvm-3.20/svm-predict
input:
      corum_train_labeled.libsvm0.scaleByTrain.txt, corum_train_labeled.libsvm1.scale.model_c32_g05
output:
      corum_train_labeled.libsvm0.scaleByTrain.resultsWprob

probability ordered list of pairs for unlabeled and test set (putting svm format into pairwise format) (order by score (prob of being true, 2nd column), dropping probability)
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/svm_results2pairs.py
inputs:
      corum_train_labeled.txt, corum_train_labeled.libsvm0.scaleByTrain.resultsWprob
output:
      corum_train_labeled.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups.txt

probability ordered list of pairs for unlabeled and test set with probability(order by score (prob of being true, 2nd column), keeping probability this time)
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/svm_results2pairs.py
inputs:
      corum_train_labeled.txt, corum_train_labeled.libsvm0.scaleByTrain.resultsWprob
output:
      corum_train_labeled.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.txt

predict training set on train model
We also want probability score for the training set
/home/kdrew/programs/libsvm-3.20/svm-predict
inputs:
      corum_train_labeled.libsvm1.scale.txt
      corum_train_labeled.libsvm1.scale.model_c32_g05
output:
      corum_train_labeled.libsvm1.scale.resultsWprob

probability ordered list of pairs for training set with probability
Order training set by probability
/home/kdrew/scripts/protein_complex_maps/protein_complex_maps/features/svm_results2pairs.py
inputs:
      corum_train_labeled.txt, corum_train_labeled.libsvm0.scaleByTrain.resultsWprob
output:
      corum_train_labeled.libsvm1.scale.resultsWprob_pairs_noself_nodups_wprob.txt

combine results from both test and training predictions (we could train everything at once so there is not this extra combination step but keeping consistent with previous work flow
cat corum_train_labeled.libsvm1.scale.resultsWprob_pairs_noself_nodups_wprob.txt corum_train_labeled.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob.txt |sort -g -k 3 -r > corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.txt

#this is a side step, not in scripts for plants
The uncover the answer (+1 or -1) and see how classifier did
--> Gives a score [0,1] where we can rank predictions
--> The evaluate in precision recall framework using the +/- 1

Cluster PPis
Plant clustering parameter sweep
Protein Interaction Network, but hairbally
At this point, want to find clusters (dense regions)

two-stage clustering:
/project/cmcwhite/protein_complex_maps/protein_complex_maps/features/clustering_parameter_optimization.py
input:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined_filtered05.txt
     /project/cmcwhite/protein_complex_maps/protein_complex_maps/orthology_proteomics/corum/nonredundant_allComplexesCore_mammals_euNOG_merged06.train.txt
outputs:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.txt
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined_filtered05.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.out

   Do a parameter sweep (about 1000 different possibilities
   PPi score threshold [1.0, 0.9., 0.8 ... .1]
   Clusterone parameters : overlap (jaccard score) [0.8, 0.7, 0.6]  -- merging complexes with overlap
                                       density (threshold of total number of interactions vs. total possible interactions) unconnected -> fully connected
   MCL inflation [1.2, 3, 4, 7]
   Process : Run through clusterone, then run clusters from clusterone through MCL.
   Output : a set of clusters times # of possible combinations
   Then select best set of clusters (usually a couple thousand)
        by comparing to corum training complex set
                -> for human optimized by clique size
        so, take a predicted complex, and see jaccard overlap with a training complex (precision)
             take a training complex, and see how many recalled (also using jaccard overlap)
             normalized based on size of the complex
        Optimized harmonic mean of precision & recall (hmean). If numbers pres/rec aren't balanced it downweights it.
            We want a balance between precision / recall
        Rank clusters by hmean, pick top one, and that's the map.
                Evaluate on the leave-out set of complexes for hmean

generate cytoscape network
(in cytoscape_networks directory)
plant complex cytoscape network

Make clusters into pairs
python /project/cmcwhite/protein_complex_maps/protein_complex_maps/util/cluster2pairwise.py
input:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.txt
output:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.pairsWclustID.txt

Make clusters into node table
/project/cmcwhite/protein_complex_maps/protein_complex_maps/util/cluster2node_table.py
input:
       corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.txt
output:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.nodeTable.txt

Make edge attribute table
/project/cmcwhite/protein_complex_maps//protein_complex_maps/util/pairwise2clusterid.py
inputs:
       corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined_filtered05.txt       corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.txt
       all .corr_poisson files
output:
      corum_train_labeled.libsvm1.scale.libsvm0.scaleByTrain.resultsWprob_pairs_noself_nodups_wprob_combined.best_cluster_wOverlap_nr_allComplexesCore_mammals_euNOG_psweep_clusterone_mcl.ii353.edgeAttributeWClusterid.txt




