/***************************************************************************
 *  Read Delphes file and save a skimmed version of root file with TLorentzvectors 
****************************************************************************/

#include <TFile.h>
#include <TTree.h>
#include <vector>
#include <iostream>
#include <TLeaf.h>


void readEventData(const char* inputFileName, const char* outputFileName, Double32_t cmsEnergy, Double32_t cross) {


    cout << "Wait.. Analysing Delphes ntuple with CMS=" << cmsEnergy << " GeV " << endl;


    // Open the ROOT file
    TFile inputFile(inputFileName);
    if (inputFile.IsZombie()) {
        std::cerr << "Error: Unable to open file " << inputFileName << std::endl;
        return;
    }

    // Access the Delphes TTree
    TTree* tree = dynamic_cast<TTree*>(inputFile.Get("Delphes"));
    if (!tree) {
        std::cerr << "Error: Unable to access tree Delphes in file " << inputFileName << std::endl;
        return;
    }
   
    // Find values for kMaxJet,kMaxElectron, kMaxMuon, kMaxPhoton
    Int_t kMaxJet = 0;
    Int_t kMaxElectron = 0;
    Int_t kMaxPhoton = 0;
    Int_t kMaxMuon = 0;
    Int_t kMaxMissingET = 0;
    //Int_t kMaxWeight = 0;
    Int_t kMaxEvent = 0;

    TBranch* jet_Pt_Branch = tree->GetBranch("Jet.PT");
    TBranch* el_Pt_Branch = tree->GetBranch("Electron.PT");
    TBranch* mu_Pt_Branch = tree->GetBranch("Muon.PT");
    TBranch* ph_Pt_Branch = tree->GetBranch("Photon.PT");
    TBranch* MET_Branch = tree->GetBranch("MissingET.MET");
    //TBranch* Weight_Branch = tree->GetBranch("Weight.Weight");
    TBranch* eventWeightBranch = tree->GetBranch("Event.Weight");

    TLeaf* jet_leaf = jet_Pt_Branch->GetLeaf("Jet.PT");
    TLeaf* el_leaf = el_Pt_Branch->GetLeaf("Electron.PT");
    TLeaf* mu_leaf = mu_Pt_Branch->GetLeaf("Muon.PT");
    TLeaf* ph_leaf = ph_Pt_Branch->GetLeaf("Photon.PT");
    TLeaf* MET_leaf = MET_Branch->GetLeaf("MissingET.MET");
    //TLeaf* Weight_leaf = Weight_Branch->GetLeaf("Weight.Weight");
    TLeaf* eventWeightLeaf = eventWeightBranch->GetLeaf("Event.Weight");

    Long64_t numEntries1 = tree->GetEntries();
    for (Long64_t i = 0; i < numEntries1; ++i) {
        jet_Pt_Branch->GetEntry(i);
        Int_t len_jet = jet_leaf->GetLen();
        if (len_jet > kMaxJet) {
            kMaxJet = len_jet;
        }
    }
    
    for (Long64_t i = 0; i < numEntries1; ++i) {
        el_Pt_Branch->GetEntry(i);
        Int_t len_el = el_leaf->GetLen();
        if (len_el > kMaxElectron) {
            kMaxElectron = len_el;
        }
    }
      
    for (Long64_t i = 0; i < numEntries1; ++i) {
        mu_Pt_Branch->GetEntry(i);
        Int_t len_mu = mu_leaf->GetLen();
        if (len_mu > kMaxMuon) {
            kMaxMuon = len_mu;
        }
    }
    
    for (Long64_t i = 0; i < numEntries1; ++i) {
        ph_Pt_Branch->GetEntry(i);
        Int_t len_ph = ph_leaf->GetLen();
        if (len_ph > kMaxPhoton) {
            kMaxPhoton = len_ph;
        }
    }

    for (Long64_t i = 0; i < numEntries1; ++i) {
        MET_Branch->GetEntry(i);
        Int_t len_MET = MET_leaf->GetLen();
        if (len_MET > kMaxMissingET) {
            kMaxMissingET = len_MET;
        }
    }

    /*
    for (Long64_t i = 0; i < numEntries1; ++i) {
        Weight_Branch->GetEntry(i);
        Int_t len_W = Weight_leaf->GetLen();
        if (len_W > kMaxWeight) {
            kMaxWeight = len_W;
        }
    }
    */

    for (Long64_t i = 0; i < numEntries1; ++i) {
        eventWeightBranch->GetEntry(i);
        Int_t len_W = eventWeightLeaf->GetLen();
        if (len_W > kMaxEvent) {
            kMaxEvent = len_W;
        }
    }
    

    std::cout << "kMaxJet: " << kMaxJet << std::endl;
    std::cout << "kMaxElectron: " << kMaxElectron << std::endl;
    std::cout << "kMaxMuon: " << kMaxMuon << std::endl;
    std::cout << "kMaxPhoton: " << kMaxPhoton << std::endl;
    std::cout << "kMaxMissingET: " << kMaxMissingET << std::endl;
    //std::cout << "kMaxWeight: " << kMaxWeight << std::endl;
    std::cout << "kMaxEvent: " << kMaxEvent << std::endl;

    Float_t         Jet_PT[kMaxJet];   //[Jet_]
    Float_t         Jet_Eta[kMaxJet];   //[Jet_]
    Float_t         Jet_Phi[kMaxJet];   //[Jet_]
    Float_t         Jet_Mass[kMaxJet];   //[Jet_]
    UInt_t          Jet_BTag[kMaxJet];   //[Jet_]

    Float_t         Electron_PT[kMaxElectron];   //[Electron_]
    Float_t         Electron_Eta[kMaxElectron];   //[Electron_]
    Float_t         Electron_Phi[kMaxElectron];   //[Electron_]
    //Float_t         Electron_Mass[kMaxElectron];   //[Electron_]

    Float_t         Muon_PT[kMaxMuon];   //[Muon_]
    Float_t         Muon_Eta[kMaxMuon];   //[Muon_]
    Float_t         Muon_Phi[kMaxMuon];   //[Muon_]
    //Float_t         Muon_Mass[kMaxMuon];   //[Muon_]

    Float_t         Photon_PT[kMaxPhoton];   //[Photon_]
    Float_t         Photon_Eta[kMaxPhoton];   //[Photon_]
    Float_t         Photon_Phi[kMaxPhoton];   //[Photon_]
    Float_t         Photon_E[kMaxPhoton];   //[Photon_]

    Float_t         MissingET_MET[kMaxMissingET];   //[MissingET_]
    Float_t         MissingET_Eta[kMaxMissingET];   //[MissingET_]
    Float_t         MissingET_Phi[kMaxMissingET];   //[MissingET_]    

    //Float_t         Weight_Weight[kMaxWeight];   //[Weight_]
    Float_t         Event_Weight[kMaxEvent];   //[Event_]

    TBranch        *b_Jet_PT;   //!
    TBranch        *b_Jet_Eta;   //!
    TBranch        *b_Jet_Phi;   //!
    TBranch        *b_Jet_T;   //!
    TBranch        *b_Jet_Mass;   //!
    TBranch        *b_Jet_BTag;   //!
    TBranch        *b_Electron_PT;   //!
    TBranch        *b_Electron_Eta;   //!
    TBranch        *b_Electron_Phi;   //!
    //TBranch        *b_Electron_Mass;   //!
    TBranch        *b_Muon_PT;   //!
    TBranch        *b_Muon_Eta;   //!
    TBranch        *b_Muon_Phi;   //!
    //TBranch        *b_Muon_Mass;   //!
    TBranch        *b_Photon_PT;   //!
    TBranch        *b_Photon_Eta;   //!
    TBranch        *b_Photon_Phi;   //!
    TBranch        *b_Photon_E;   //!

    TBranch        *b_MissingET_MET;   //!
    TBranch        *b_MissingET_Eta;   //!
    TBranch        *b_MissingET_Phi;   //!

    //TBranch        *b_Weight_Weight;   //!
    TBranch        *b_Event_Weight;   //!
    
    tree->SetBranchAddress("Jet.PT", Jet_PT, &b_Jet_PT);
    tree->SetBranchAddress("Jet.Eta", Jet_Eta, &b_Jet_Eta);
    tree->SetBranchAddress("Jet.Phi", Jet_Phi, &b_Jet_Phi);
    tree->SetBranchAddress("Jet.Mass", Jet_Mass, &b_Jet_Mass);
    tree->SetBranchAddress("Jet.BTag", Jet_BTag, &b_Jet_BTag);
    
    tree->SetBranchAddress("Electron.PT", Electron_PT, &b_Electron_PT);
    tree->SetBranchAddress("Electron.Eta", Electron_Eta, &b_Electron_Eta);
    tree->SetBranchAddress("Electron.Phi", Electron_Phi, &b_Electron_Phi);
    //tree->SetBranchAddress("Electron.Mass", Electron_Mass, &b_Electron_Mass);

    tree->SetBranchAddress("Muon.PT", Muon_PT, &b_Muon_PT);
    tree->SetBranchAddress("Muon.Eta", Muon_Eta, &b_Muon_Eta);
    tree->SetBranchAddress("Muon.Phi", Muon_Phi, &b_Muon_Phi);
    //tree->SetBranchAddress("Muon.Mass", Muon_Mass, &b_Muon_Mass);

    tree->SetBranchAddress("Photon.PT", Photon_PT, &b_Photon_PT);
    tree->SetBranchAddress("Photon.Eta", Photon_Eta, &b_Photon_Eta);
    tree->SetBranchAddress("Photon.Phi", Photon_Phi, &b_Photon_Phi);
    tree->SetBranchAddress("Photon.E", Photon_E, &b_Photon_E);

    tree->SetBranchAddress("MissingET.Eta", MissingET_Eta, &b_MissingET_Eta);
    tree->SetBranchAddress("MissingET.Phi", MissingET_Phi, &b_MissingET_Phi);
    tree->SetBranchAddress("MissingET.MET", MissingET_MET, &b_MissingET_MET);

    //tree->SetBranchAddress("Weight.Weight", Weight_Weight, &b_Weight_Weight);
    tree->SetBranchAddress("Event.Weight", Event_Weight, &b_Event_Weight);

    Long64_t numEntries = tree->GetEntries();
    cout << "no. of events :" << numEntries << endl;

    std::vector<Double32_t> bJET_pt, bJET_eta, bJET_phi, bJET_mass;
    Int_t  N_bJET;

    std::vector<Double32_t> JET_pt, JET_eta, JET_phi, JET_mass;
    Int_t N_JET;

    std::vector<Double32_t> EL_pt, EL_eta, EL_phi;
    Int_t N_EL;

    std::vector<Double32_t> MU_pt, MU_eta, MU_phi;
    Int_t N_MU;

    std::vector<Double32_t> PH_pt, PH_eta, PH_phi, PH_e;
    Int_t N_PH;

    std::vector<Double32_t> MET_eta, MET_phi, MET_met;

    std::vector<Double32_t> Evt_Weight;

    TFile outputFile(outputFileName, "RECREATE");
    TTree ntuple("Ntuple", "Ntuple for ADFilter");

    // jet branches
    ntuple.Branch("JET_n",  &N_JET);
    ntuple.Branch("JET_pt", &JET_pt, 256000, 0);
    ntuple.Branch("JET_eta", &JET_eta, 256000, 0);
    ntuple.Branch("JET_phi", &JET_phi, 256000, 0);
    ntuple.Branch("JET_mass", &JET_mass, 256000, 0);

    // b jet branches
    ntuple.Branch("bJET_n",  &N_bJET);
    ntuple.Branch("bJET_pt", &bJET_pt, 256000, 0);
    ntuple.Branch("bJET_eta", &bJET_eta, 256000, 0);
    ntuple.Branch("bJET_phi", &bJET_phi, 256000, 0);
    ntuple.Branch("bJET_mass", &bJET_mass, 256000, 0);
    
    // Electron branches
    ntuple.Branch("EL_n", &N_EL);
    ntuple.Branch("EL_pt", &EL_pt, 256000, 0);
    ntuple.Branch("EL_eta", &EL_eta, 256000, 0);
    ntuple.Branch("EL_phi", &EL_phi, 256000, 0);

    // Muon branches
    ntuple.Branch("MU_n", &N_MU);
    ntuple.Branch("MU_pt", &MU_pt, 256000, 0);
    ntuple.Branch("MU_eta", &MU_eta, 256000, 0);
    ntuple.Branch("MU_phi", &MU_phi, 256000, 0);

    // Photon branches
    ntuple.Branch("PH_n", &N_PH);
    ntuple.Branch("PH_pt", &PH_pt, 256000, 0);
    ntuple.Branch("PH_eta", &PH_eta, 256000, 0);
    ntuple.Branch("PH_phi", &PH_phi, 256000, 0);
    ntuple.Branch("PH_e", &PH_e, 256000, 0);

    // MissingET branches
    ntuple.Branch("MET_eta", &MET_eta, 256000, 0);
    ntuple.Branch("MET_phi", &MET_phi, 256000, 0);
    ntuple.Branch("MET_met", &MET_met, 256000, 0);
    
    ntuple.Branch("Evt_Weight", &Evt_Weight, 256000, 0);

    TH1F* meta = new TH1F("meta","meta",10,0,10);
    meta->Fill("CMS energy [GeV]", cmsEnergy);
    meta->Fill("cross section [PB]", cross);
    meta->SetBinError(1,0);
    meta->SetBinError(2,0);
    meta->Write();

    for (Long64_t i = 0; i < numEntries; ++i) {
        // Clear the vector for the current event
        //jet_Vectors.clear();

        JET_pt.clear(); 
        JET_eta.clear();
        JET_phi.clear();
        JET_mass.clear();
        N_JET=0;
        
        bJET_pt.clear();
        bJET_eta.clear();
        bJET_phi.clear();
        bJET_mass.clear();
        N_bJET=0;
        
        EL_pt.clear();
        EL_eta.clear();
        EL_phi.clear();
        N_EL = 0;

        MU_pt.clear();
        MU_eta.clear();
        MU_phi.clear();
        N_MU = 0;

        PH_pt.clear();
        PH_eta.clear();
        PH_phi.clear();
        PH_e.clear();
        N_PH = 0;

        MET_eta.clear();
        MET_phi.clear();
        MET_met.clear();
        
        Evt_Weight.clear();

        // Get entry i
        tree->GetEntry(i);

       if (i<=10 &&
       (i<=100 && (i%10) == 0) ||
       (i<=10000 && (i%1000) == 0)  ||
       (i>=10000 && (i%10000) == 0) ) {
          cout << "Event # " << i << endl; }


        //cout << " " << endl;

        TLeaf* jet_leaf = jet_Pt_Branch->GetLeaf("Jet.PT");
        TLeaf* el_leaf = el_Pt_Branch->GetLeaf("Electron.PT");
        TLeaf* mu_leaf = mu_Pt_Branch->GetLeaf("Muon.PT");
        TLeaf* ph_leaf = ph_Pt_Branch->GetLeaf("Photon.PT");
        TLeaf* MET_leaf = MET_Branch->GetLeaf("MissingET.MET");
        TLeaf* eventWeightLeaf = eventWeightBranch->GetLeaf("Event.Weight");

        Long64_t len_jet = jet_leaf->GetLen();
        Long64_t len_el = el_leaf->GetLen();
        Long64_t len_mu = mu_leaf->GetLen();
        Long64_t len_ph = ph_leaf->GetLen();
        Long64_t len_MET = MET_leaf->GetLen();
        Long64_t len_W = eventWeightLeaf->GetLen();


        // fill light jets and b-jets 
        for (Int_t j = 0; j < len_jet; ++j) {
            if ((Jet_BTag[j]) == 1) { 
             //cout << "bjet : " << j << " " << Jet_PT[j] << endl;
              bJET_pt.push_back(Jet_PT[j]);
              bJET_eta.push_back(Jet_Eta[j]);
              bJET_phi.push_back(Jet_Phi[j]);
              bJET_mass.push_back(Jet_Mass[j]);
            } else {
              JET_pt.push_back(Jet_PT[j]);
              JET_eta.push_back(Jet_Eta[j]);
              JET_phi.push_back(Jet_Phi[j]);
              JET_mass.push_back(Jet_Mass[j]);
         }
        }
        
        // Electron data
        for (Int_t j = 0; j < len_el; ++j) {
            //cout << "electron: " << j << " " << Electron_PT[j] << endl;
            EL_pt.push_back(Electron_PT[j]);
            EL_eta.push_back(Electron_Eta[j]);
            EL_phi.push_back(Electron_Phi[j]);
        }

        // Muon data
        for (Int_t j = 0; j < len_mu; ++j) {
            //cout << "muon: " << j << " " << Muon_PT[j] << endl;
            MU_pt.push_back(Muon_PT[j]);
            MU_eta.push_back(Muon_Eta[j]);
            MU_phi.push_back(Muon_Phi[j]);
        }

        // Photon data
        for (Int_t j = 0; j < len_ph; ++j) {
            //cout << "photon: " << j << " " << Photon_PT[j] << endl;
            PH_pt.push_back(Photon_PT[j]);
            PH_eta.push_back(Photon_Eta[j]);
            PH_phi.push_back(Photon_Phi[j]);
            PH_e.push_back(Photon_E[j]);
        }

        // MissingET data
        for (Int_t j = 0; j < len_MET; ++j) {
            //cout << "missingET: " << j << " " << MissingET_MET[j] << endl;
            MET_eta.push_back(MissingET_Eta[j]);
            MET_phi.push_back(MissingET_Phi[j]);
            MET_met.push_back(MissingET_MET[j]);
        }
        
        // Event.Weight data
        for (Int_t j = 0; j < len_W; ++j) {
            //cout << "Event.Weight: " << j << " " <<  Event_Weight[j] << endl;
            Evt_Weight.push_back(Event_Weight[j]);
        }
     

       N_JET=JET_pt.size();
       N_bJET=bJET_pt.size();
       N_EL = EL_pt.size();
       N_MU = MU_pt.size();
       N_PH = PH_pt.size();
 
       // fill ntuples
       ntuple.Fill();

    }

     // Close the file
    inputFile.Close();

   // Write the TTree to the output file
    ntuple.Write();
 
    // Close the output file
    outputFile.Close();
    
    
}

void skim_Delphes(const char* inputFileName = nullptr, const char* outputFileName = nullptr, Double32_t cmsEnergy = 0.0, Double32_t cross = 0.0) {
    // Check if both input and output file names are provided
    if (!inputFileName || !outputFileName) {
        std::cerr << "Usage: root -b -q skim_Delphes.C(\"inputFileName\", \"outputFileName\", cmsEnergy, cross)" << std::endl;
        return;
    }
    // Call the skim_Delphes_file function with the provided file names
    readEventData(inputFileName, outputFileName, cmsEnergy, cross);
}


