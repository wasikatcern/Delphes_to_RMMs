# The code converts particle collision events from Delphes files into Rapidity Mass Matrices(RMMs)
#RMM Reference: "Imaging particle collision data for event classification using machine learning", by S.V. Chekanov, doi = 10.1016/j.nima.2019.04.031

#Convert Delphes file into a skimmed version with relevant information as needed for RMMs. You can specify center of mass energy (say 13000 GeV) and the cross section for which you are scaling it (say 100 pb for example). These skimmed files can also be used in the ADFilter website (https://mc.hep.anl.gov/adfilter).

root -l -b 'skim_Delphes.C("Hplus_1800GeV_SLHA2_delphes.root", "skimmed_delphes.root", 13000, 100)'

#Convert skimmed delphes file into RMMs; Specify how many number of events you want to process, for 100 events :

root -l -b -q 'make_RMMs.C("skimmed_delphes.root","rmm_events_100.csv",100)'

#Specify event number to plot (event# 12 for example below); Inside code you can specify no. of objects you want to plot in matrix

python plot_rmm.py --csv rmm_events_100.csv --event 12
~                                                                                                                                                           
~                                                                                                                                                           
~                                                                                                                                                           
~                                                                                                                                                           
~       
