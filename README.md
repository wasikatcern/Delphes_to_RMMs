# Convert particle collision events from Delphes files into Rapidity Mass Matrices(RMMs)
#RMM Reference: "Imaging particle collision data for event classification using machine learning", by S.V. Chekanov, https://doi.org/10.1016/j.nima.2019.04.031 .

#Convert Delphes file into a skimmed version with relevant information as needed for RMMs. You can specify the center of mass energy (for example, 13000 GeV) and the cross section for which you are scaling it (say 100 pb, for example, as shown below). These skimmed files can also be used in the ADFilter website (https://mc.hep.anl.gov/adfilter).

root -l -b 'skim_Delphes.C("Hplus_1800GeV_SLHA2_delphes.root", "skimmed_delphes.root", 13000, 100)'

#Convert skimmed Delphes file into RMMs; Specify how many events you want to process. For all events use -1 (as shown below), for 100 events use 100 instead of -1 :

root -l -b -q 'make_RMMs.C("skimmed_delphes.root","rmm_events_100.csv",-1)'

#Specify event number to plot (event# 12 is used below); Inside code, you can specify the number of different objects you want to plot in a matrix

python plot_rmm.py --csv rmm_events_100.csv --event 12
