/***************************************************************************************************************
 * Prepare RMM matrices from skimmed root files after processing from Delphes/other formats &
 * save as CSV file + ALSO write a "compact-20" CSV computed from the RMM (à la rmm_compact20.py)
 *
 * Examples:
 *   root -l -b -q 'make_RMMs_compact20.C("skimmed_delphes.root","rmm_events_100.csv",100)'
 *   root -l -b -q 'make_RMMs_compact20.C' --args --in skimmed_delphes.root --out rmm_events_100.csv --nevents 100
 *   root -l -b -q 'make_RMMs_compact20.C' --args --in skimmed_delphes.root --out rmm.csv --out20 rmm_compact20.csv --selroot my_selected.root --iconfig 0
 *
 * This file is adapted from make_RMMs.C and adds compact-20 output. (See original for details.)
 **************************************************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sstream>

#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TApplication.h>
#include <TSystem.h>
#include <TMath.h>
#include <Rtypes.h>

using namespace std;

// ---------------------- CParticle.h -------------------------
class CParticle : public TObject {
protected:
  int  Px,Py,Pz,E,Mass;
  int  Charge;
  int  m_pid;
  int  m_status;
  std::vector<int>  parameters;

public:
  CParticle()
  : Px(0),Py(0),Pz(0),E(0),Mass(0),Charge(0),m_pid(0),m_status(0),parameters() {}

  CParticle(int pX, int pY, int pZ )
  : Px(pX), Py(pY), Pz(pZ), E(0), Mass(0), Charge(0), m_pid(0), m_status(0) { parameters.clear(); }

  CParticle(int pX, int pY, int pZ, int charge )
  : Px(pX), Py(pY), Pz(pZ), E(0), Mass(0), Charge(charge), m_pid(0), m_status(0) { parameters.clear(); }

  CParticle(int pX, int pY, int pZ, int mass, int charge )
  : Px(pX), Py(pY), Pz(pZ), E(0), Mass(mass), Charge(charge), m_pid(0), m_status(0) { parameters.clear(); }

  CParticle(int pX, int pY, int pZ, int mass, int charge, int energy )
  : Px(pX), Py(pY), Pz(pZ), E(energy), Mass(mass), Charge(charge), m_pid(0), m_status(0) { parameters.clear(); }

  virtual ~CParticle() {}

  CParticle(CParticle* p){
    if(p){
      Px=p->Px; Py=p->Py; Pz=p->Pz; E=p->E; Mass=p->Mass; Charge=p->Charge;
      m_pid=p->m_pid; m_status=p->m_status; parameters=p->parameters;
    }else{
      Px=Py=Pz=E=Mass=Charge=m_pid=m_status=0; parameters.clear();
    }
  }

  void setParameter(int q) { parameters.push_back(q); }
  std::vector<int> getParameters() const { return parameters; }

  // get methods
  int  px() const { return Px; }
  int  py() const { return Py; }
  int  pz() const { return Pz; }
  int  e()  const { return E;  }
  int  id() const { return m_pid; }
  void getPxPyPz(int &px, int &py, int &pz) const { px=Px; py=Py; pz=Pz; }
  int  mass()   const { return Mass; }
  int  charge() const { return Charge; }
  int  status() const { return m_status; }

  // - set values
  void setID(int c) { m_pid=c; }
  void setCharge(int c) { Charge=c; }
  void setStatus(int c) { m_status=c; }
  void setMass(int  m)  { Mass=m; }
  void setE(int e_) { E=e_; }

  // invariant mass helpers (not used below)
  double invMassE() const {
    double p2 = (double)Px*Px + (double)Py*Py + (double)Pz*Pz;
    double m2 = (double)E*E - p2;
    return (m2>0.0)? std::sqrt(m2) : 0.0;
  }
  double invMassM() const { return (double)Mass; }

  // calculated values
  double  calcPt() const { return TMath::Sqrt((double)Px*(double)Px+(double)Py*(double)Py); }
  double  calcPhi() const { return TMath::ATan2((double)Py,(double)Px); }
  double  calcPhiWrap() const { double phi = calcPhi(); if (phi<0) phi+=2.0*TMath::Pi(); return phi; }
  double  calcPhiWrapGrad() const { return calcPhiWrap() * 180.0 / TMath::Pi(); }
  double  calcP()  const { return TMath::Sqrt((double)Px*(double)Px+(double)Py*(double)Py+(double)Pz*(double)Pz); }
  double  calcE()  const { return TMath::Sqrt((double)Px*(double)Px+(double)Py*(double)Py+(double)Pz*(double)Pz+(double)Mass*(double)Mass); }
  double  calcEta() const {
    double p = calcP(), denom = p - (double)Pz, numer = p + (double)Pz;
    if (denom <= 0 || numer <= 0) { if (Pz>0) return 10.0; if (Pz<0) return -10.0; return 0.0; }
    return 0.5 * std::log(numer/denom);
  }

  CParticle copy() const {
    CParticle q(Px,Py,Pz,Mass,Charge,E);
    q.setID(m_pid); q.setStatus(m_status);
    for (auto v: parameters) q.setParameter(v);
    return q;
  }

  CParticle operator+(CParticle p) const {
    CParticle out(Px + p.Px, Py + p.Py, Pz + p.Pz, Mass + p.Mass, Charge + p.Charge, E + p.E);
    return out;
  }

  bool operator<( const CParticle& rhs ) const {
    return (double)Px*(double)Px + (double)Py*(double)Py > rhs.Px*rhs.Px + rhs.Py*rhs.Py;
  }

  void print(){ std::cout << toString() << std::endl; }
  string toString() const {
    std::ostringstream ss;
    ss << "CParticle(px="<<Px<<", py="<<Py<<", pz="<<Pz
       <<", E="<<E<<", M="<<Mass<<", Q="<<Charge
       <<", id="<<m_pid<<", status="<<m_status<<")";
    return ss.str();
  }

  ClassDef(CParticle,1);
};

// ---------------------- LParticle.h -------------------------
class LParticle: public TObject {
private:
  Int_t m_type;
  Int_t m_status;
  Int_t m_charge;
  Int_t m_parent;
  TLorentzVector momentum;
  std::vector<double>  parameters;
  std::vector<CParticle>  constituents;

public:
  LParticle()
  : m_type(0), m_status(0), m_charge(0), m_parent(-1), momentum(0,0,0,0),
    parameters(), constituents() {}

  LParticle(Double_t px, Double_t py, Double_t pz, Double_t e, Int_t charge)
  : m_type(0), m_status(0), m_charge(charge), m_parent(-1),
    momentum(px,py,pz,e), parameters(), constituents() {}

  LParticle(LParticle* p){
    if(p){
      m_type=p->m_type; m_status=p->m_status; m_charge=p->m_charge; m_parent=p->m_parent;
      momentum=p->momentum; parameters=p->parameters; constituents=p->constituents;
    }else{
      m_type=0; m_status=0; m_charge=0; m_parent=-1; momentum.SetPxPyPzE(0,0,0,0);
      parameters.clear(); constituents.clear();
    }
  }

  LParticle(Int_t charge)
  : m_type(0), m_status(0), m_charge(charge), m_parent(-1), momentum(0,0,0,0),
    parameters(), constituents() {}

  ~LParticle() {}

  Int_t GetType()   const { return m_type;    }
  Int_t GetStatus() const { return m_status;  }
  Int_t GetParent() const { return m_parent;  }
  Int_t GetCharge() const { return m_charge;  }
  TLorentzVector GetP() const { return momentum; }

  void SetCharge(Int_t q) { m_charge = q; }
  void SetParent(Int_t q) { m_parent = q; }
  void SetType(Int_t q)   { m_type = q;   }
  void SetStatus(Int_t q) { m_status = q; }
  void SetP(const TLorentzVector& mom){ momentum = mom; }
  void SetParameter(double q) { parameters.push_back(q); }
  void SetConstituent(CParticle q) { constituents.push_back(q); }

  std::vector<double>    GetParameters()    const { return parameters; }
  std::vector<CParticle> GetConstituents()  const { return constituents; }

  bool operator> (const LParticle& i) const {
    return (momentum.Pt() > i.momentum.Pt());
  }

  ClassDef(LParticle,1)
};

ClassImp(CParticle)
ClassImp(LParticle)

// ---------- CLI options ----------
struct Opts {
  string in, out;
  string out20;              // NEW: output CSV for compact-20
  string selroot;            // selected-events ROOT file
  int maxNumber = 10;
  int maxTypes  = 5;         // jets,bjets,mu,el,ph (+1 row/col is MET)
  int iconfig   = 0;
  long long nevents = -1;
  double ptJet = 30.0, ptLep = 30.0, etaJet = 2.4, etaLep = 2.4, leadLepPt = 60.0;
};

static bool parse_args(int argc, char** argv, Opts& o){
  for (int i=1;i<argc;i++){
    string a = argv[i];
    auto need=[&](int i){ if(i+1>=argc){ cerr<<"Missing value for "<<a<<"\n"; return false;} return true; };
    if      (a=="--in"){ if(!need(i))return false; o.in=argv[++i]; }
    else if (a=="--out"){ if(!need(i))return false; o.out=argv[++i]; }
    else if (a=="--out20"){ if(!need(i))return false; o.out20=argv[++i]; }              // NEW
    else if (a=="--selroot"){ if(!need(i))return false; o.selroot=argv[++i]; }
    else if (a=="--maxN"){ if(!need(i))return false; o.maxNumber=atoi(argv[++i]); }
    else if (a=="--types"){ if(!need(i))return false; o.maxTypes=atoi(argv[++i]); }
    else if (a=="--iconfig"){ if(!need(i))return false; o.iconfig=atoi(argv[++i]); }
    else if (a=="--nevents"){ if(!need(i))return false; o.nevents=atoll(argv[++i]); }
    else if (a=="--ptJet"){ if(!need(i))return false; o.ptJet=atof(argv[++i]); }
    else if (a=="--ptLep"){ if(!need(i))return false; o.ptLep=atof(argv[++i]); }
    else if (a=="--etaJet"){ if(!need(i))return false; o.etaJet=atof(argv[++i]); }
    else if (a=="--etaLep"){ if(!need(i))return false; o.etaLep=atof(argv[++i]); }
    else if (a=="--leadLepPt"){ if(!need(i))return false; o.leadLepPt=atof(argv[++i]); }
    else { cerr<<"Unknown arg: "<<a<<"\n"; return false; }
  }
  if (o.in.empty() || o.out.empty()){
    cerr<<"Usage:\n"
        <<"  root -l -b -q 'make_RMMs_compact20.C(\"in.root\",\"out.csv\",100)'\n"
        <<"  root -l -b -q 'make_RMMs_compact20.C' --args --in in.root --out out.csv --nevents 100 [--out20 out_compact20.csv --selroot out_selected.root]\n";
    return false;
  }
  return true;
}

// ---------- Selection helpers ----------
static bool select_event(const Opts& opt, double /*met_pt*/,
                         const vector<LParticle>& mu, const vector<LParticle>& el,
                         const vector<LParticle>& alljets)
{
  int k = opt.iconfig;
  if (k==-1) return true;
  if (k==0 || k==2){
    if (!mu.empty() && mu[0].GetP().Perp()>opt.leadLepPt) return true;
    if (!el.empty() && el[0].GetP().Perp()>opt.leadLepPt) return true;
    return false;
  }
  if (k==1) return true;
  if (k==3){
    int n=0;
    for (const auto& p:mu) if (p.GetP().Perp()>25.0) n++;
    for (const auto& p:el) if (p.GetP().Perp()>25.0) n++;
    return n>1;
  }
  if (k==6)    return (!alljets.empty() && alljets[0].GetP().Perp()>500.0);
  if (k==1000) return (!alljets.empty() && alljets[0].GetP().Perp()>30.0);
  return true;
}
static void sort_by_pt(vector<LParticle>& v){
  sort(v.begin(), v.end(),
       [](const LParticle& a, const LParticle& b)->bool{
         return a.GetP().Perp() > b.GetP().Perp();
       });
}

// ---------- Bind vector-leaf schema ----------
struct VecLeaves {
  Int_t JET_n=0,bJET_n=0,EL_n=0,MU_n=0,PH_n=0;
  vector<double>* JET_pt=nullptr;  vector<double>* JET_eta=nullptr;  vector<double>* JET_phi=nullptr;  vector<double>* JET_mass=nullptr;
  vector<double>* bJET_pt=nullptr; vector<double>* bJET_eta=nullptr; vector<double>* bJET_phi=nullptr; vector<double>* bJET_mass=nullptr;
  vector<double>* EL_pt=nullptr; vector<double>* EL_eta=nullptr; vector<double>* EL_phi=nullptr;
  vector<double>* MU_pt=nullptr; vector<double>* MU_eta=nullptr; vector<double>* MU_phi=nullptr;
  vector<double>* PH_pt=nullptr; vector<double>* PH_eta=nullptr; vector<double>* PH_phi=nullptr;
  vector<double>* MET_met=nullptr; vector<double>* MET_phi=nullptr;
  vector<double>* Evt_Weight=nullptr;
};
static bool bind_schema_vec(TTree* nt, VecLeaves& v){
  auto need=[&](const char* b)->bool{ return nt->GetBranch(b)!=nullptr; };
  bool ok=true;
  ok &= need("JET_pt")&&need("JET_eta")&&need("JET_phi")&&need("JET_mass");
  ok &= need("bJET_pt")&&need("bJET_eta")&&need("bJET_phi")&&need("bJET_mass");
  ok &= need("EL_pt")&&need("EL_eta")&&need("EL_phi");
  ok &= need("MU_pt")&&need("MU_eta")&&need("MU_phi");
  ok &= need("PH_pt")&&need("PH_eta")&&need("PH_phi");
  ok &= need("MET_met")&&need("MET_phi");
  ok &= need("Evt_Weight");
  if (!ok){ cerr<<"ERROR: missing required branches.\n"; return false; }

  if (nt->GetBranch("JET_n"))  nt->SetBranchAddress("JET_n",  &v.JET_n);
  if (nt->GetBranch("bJET_n")) nt->SetBranchAddress("bJET_n", &v.bJET_n);
  if (nt->GetBranch("EL_n"))   nt->SetBranchAddress("EL_n",   &v.EL_n);
  if (nt->GetBranch("MU_n"))   nt->SetBranchAddress("MU_n",   &v.MU_n);
  if (nt->GetBranch("PH_n"))   nt->SetBranchAddress("PH_n",   &v.PH_n);

  nt->SetBranchAddress("JET_pt",&v.JET_pt);   nt->SetBranchAddress("JET_eta",&v.JET_eta);
  nt->SetBranchAddress("JET_phi",&v.JET_phi); nt->SetBranchAddress("JET_mass",&v.JET_mass);
  nt->SetBranchAddress("bJET_pt",&v.bJET_pt); nt->SetBranchAddress("bJET_eta",&v.bJET_eta);
  nt->SetBranchAddress("bJET_phi",&v.bJET_phi); nt->SetBranchAddress("bJET_mass",&v.bJET_mass);
  nt->SetBranchAddress("EL_pt",&v.EL_pt);     nt->SetBranchAddress("EL_eta",&v.EL_eta); nt->SetBranchAddress("EL_phi",&v.EL_phi);
  nt->SetBranchAddress("MU_pt",&v.MU_pt);     nt->SetBranchAddress("MU_eta",&v.MU_eta); nt->SetBranchAddress("MU_phi",&v.MU_phi);
  nt->SetBranchAddress("PH_pt",&v.PH_pt);     nt->SetBranchAddress("PH_eta",&v.PH_eta); nt->SetBranchAddress("PH_phi",&v.PH_phi);
  nt->SetBranchAddress("MET_met",&v.MET_met); nt->SetBranchAddress("MET_phi",&v.MET_phi);
  nt->SetBranchAddress("Evt_Weight",&v.Evt_Weight);
  return true;
}

// ---------- Helpers ----------
static float getAngle(const float /*CMS*/, const TLorentzVector p1, const TLorentzVector p2){
  double y1=p1.Rapidity(), y2=p2.Rapidity();
  double HL = TMath::CosH(0.5*(y2-y1)) - 1.0;
  return (float)HL;
}
static float getMass(const float CMS, const TLorentzVector p1, const TLorentzVector p2){
  TLorentzVector pp=p1+p2;
  return (float)(pp.M()/CMS);
}
static float getHL(const TLorentzVector p1){
  double y=p1.Rapidity();
  double HL=TMath::CosH(y)-1.0;
  return (float)HL;
}
static float getMT(const TLorentzVector met, const TLorentzVector jet){
  double ss = (jet.Et()+met.Et())*(jet.Et()+met.Et())
            - (jet.Px()+met.Px())*(jet.Px()+met.Px())
            - (jet.Py()+met.Py())*(jet.Py()+met.Py());
  return (float)((ss>0)? TMath::Sqrt(ss) : 0.0);
}

// ---------- RMM projector ----------
static float** projectevent(const float CMS, const int maxN, const int maxNumberTypes,
                            const vector<LParticle> missing,
                            const vector<LParticle> jets,
                            const vector<LParticle> bjets,
                            const vector<LParticle> muons,
                            const vector<LParticle> electrons,
                            const vector<LParticle> photons)
{
  const int maxNumber = maxN;
  const int maxTypes  = maxNumberTypes;
  const int maxSize   = maxNumber*maxTypes + 1;
  const int height=maxSize, width=maxSize;

  float** outMatrix = new float*[height];
  for (int h=0; h<height; ++h){ outMatrix[h] = new float[width]; std::fill(outMatrix[h], outMatrix[h]+width, 0.0f); }

  // MET (0,0)
  TLorentzVector LMET; if (!missing.empty()) LMET = missing[0].GetP();
  outMatrix[0][0] = (float)(LMET.Et()/CMS);

  auto fill_block = [&](const vector<LParticle>& v, unsigned INCR){
    unsigned m = std::min<unsigned>(maxNumber, v.size());
    for (unsigned k1=0;k1<m;k1++){
      TLorentzVector p1 = v[k1].GetP();
      if (LMET.Et()>0) outMatrix[0][k1+INCR*maxNumber+1] = getMT(LMET,p1)/CMS;
      outMatrix[k1+INCR*maxNumber+1][0] = getHL(p1);

      if (k1==0){
        outMatrix[k1+INCR*maxNumber+1][k1+INCR*maxNumber+1] = (float)(p1.Et()/CMS);
      }else{
        TLorentzVector pprev = v[k1-1].GetP();
        float imbalance = (pprev.Et()-p1.Et())/(pprev.Et()+p1.Et());
        outMatrix[k1+INCR*maxNumber+1][k1+INCR*maxNumber+1] = imbalance;
      }
      for (unsigned k2=0;k2<m;k2++){
        TLorentzVector p2 = v[k2].GetP();
        if (k1<k2) outMatrix[k1+INCR*maxNumber+1][k2+INCR*maxNumber+1] = getMass(CMS,p1,p2);
        if (k1>k2) outMatrix[k1+INCR*maxNumber+1][k2+INCR*maxNumber+1] = getAngle(CMS,p1,p2);
      }
    }
    return m;
  };

  unsigned mJets = fill_block(jets,0);
  unsigned mB    = fill_block(bjets,1);
  unsigned mMu   = fill_block(muons,2);
  unsigned mEl   = fill_block(electrons,3);
  unsigned mPh   = fill_block(photons,4);

  auto pair_blocks = [&](const vector<LParticle>& A, unsigned INCR_A,
                         const vector<LParticle>& B, unsigned INCR_B,
                         unsigned mA, unsigned mB){
    for (unsigned i=0;i<mA;i++){
      TLorentzVector pA = A[i].GetP();
      for (unsigned j=0;j<mB;j++){
        TLorentzVector pB = B[j].GetP();
        outMatrix[i+INCR_A*maxNumber+1][j+INCR_B*maxNumber+1] = getMass(CMS,pA,pB);
        outMatrix[j+INCR_B*maxNumber+1][i+INCR_A*maxNumber+1] = getAngle(CMS,pA,pB);
      }
    }
  };
  pair_blocks(jets,0,bjets,1,mJets,mB);
  pair_blocks(jets,0,muons,2,mJets,mMu);
  pair_blocks(jets,0,electrons,3,mJets,mEl);
  pair_blocks(jets,0,photons,4,mJets,mPh);
  pair_blocks(bjets,1,muons,2,mB,mMu);
  pair_blocks(bjets,1,electrons,3,mB,mEl);
  pair_blocks(bjets,1,photons,4,mB,mPh);
  pair_blocks(muons,2,electrons,3,mMu,mEl);
  pair_blocks(muons,2,photons,4,mMu,mPh);
  pair_blocks(electrons,3,photons,4,mEl,mPh);

  return outMatrix;
}

// ---------- Compact-20 helpers (mirroring rmm_compact20.py logic) ----------
static vector<string> type_names_default(){
  return {"Jets","bJets","Muons","Electrons","Photons"};
}
/*
static vector<string> compact20_labels(int maxTypes){
  auto names = type_names_default();
  if ((int)names.size() > maxTypes) names.resize(maxTypes);
  vector<string> labels;

  // pair/type labels (i <= j): size = maxTypes*(maxTypes+1)/2
  for (int i=0;i<maxTypes;i++){
    for (int j=i;j<maxTypes;j++){
      labels.push_back(names[i] + u8"↔" + names[j]);
    }
  }
  // MET↔type labels: size = maxTypes
  for (int i=0;i<maxTypes;i++){
    labels.push_back(string("MET") + u8"↔" + names[i]);
  }
  return labels; // total = maxTypes*(maxTypes+1)/2 + maxTypes (e.g., 5->20)
}
*/
static std::vector<std::string> compact20_labels(int maxTypes){
  auto names = type_names_default();
  if ((int)names.size() > maxTypes) names.resize(maxTypes);
  std::vector<std::string> labels;

  const std::string SEP = "<->";  // ASCII separator (portable on lxplus/ROOT)

  // pair/type labels (i <= j): size = maxTypes*(maxTypes+1)/2
  for (int i=0;i<maxTypes;i++){
    for (int j=i;j<maxTypes;j++){
      labels.push_back(names[i] + SEP + names[j]);
    }
  }
  // MET↔type labels (ASCII arrow replaced by SEP): size = maxTypes
  for (int i=0;i<maxTypes;i++){
    labels.push_back(std::string("MET") + SEP + names[i]);
  }
  return labels; // total = maxTypes*(maxTypes+1)/2 + maxTypes (e.g., 5->20)
}


static double block_frobenius(const vector<float>& dense, int mSize,
                              int r0, int r1, int c0, int c1){
  // sum_{r=r0..r1-1} sum_{c=c0..c1-1} A[r,c]^2
  long double acc = 0.0;
  for (int r=r0; r<r1; ++r){
    const int base = r*mSize;
    for (int c=c0; c<c1; ++c){
      long double v = dense[base + c];
      acc += v*v;
    }
  }
  return std::sqrt((double)acc);
}

static double row_segment_norm(const vector<float>& dense, int mSize,
                               int row, int c0, int c1){
  long double acc = 0.0;
  const int base = row*mSize;
  for (int c=c0; c<c1; ++c){ long double v = dense[base + c]; acc += v*v; }
  return std::sqrt((double)acc);
}

static double col_segment_norm(const vector<float>& dense, int mSize,
                               int col, int r0, int r1){
  long double acc = 0.0;
  for (int r=r0; r<r1; ++r){ long double v = dense[r*mSize + col]; acc += v*v; }
  return std::sqrt((double)acc);
}

static vector<double> compact20_values_from_matrix(const vector<float>& dense,
                                                   int maxTypes, int maxNumber){
  const int mSize = maxTypes*maxNumber + 1; // includes MET at 0
  vector<double> vals; vals.reserve(maxTypes*(maxTypes+1)/2 + maxTypes);

  // Slices for each type block
  struct Slice { int start; int stop; };
  vector<Slice> S(maxTypes);
  for (int t=0; t<maxTypes; ++t){
    int start = 1 + t*maxNumber;
    S[t] = {start, start + maxNumber}; // [start, stop)
  }

  // Pair/type Frobenius norms: diag direct; off-diag via sqrt(||A_ij||^2 + ||A_ji||^2)
  for (int i=0;i<maxTypes;i++){
    for (int j=i;j<maxTypes;j++){
      if (i==j){
        vals.push_back(block_frobenius(dense, mSize, S[i].start, S[i].stop, S[j].start, S[j].stop));
      }else{
        double a = block_frobenius(dense, mSize, S[i].start, S[i].stop, S[j].start, S[j].stop);
        double b = block_frobenius(dense, mSize, S[j].start, S[j].stop, S[i].start, S[i].stop);
        vals.push_back(std::sqrt(a*a + b*b));
      }
    }
  }

  // MET couplings: sqrt( ||row 0, slice||^2 + ||col slice, 0||^2 )
  for (int i=0;i<maxTypes;i++){
    double a = row_segment_norm(dense, mSize, /*row=*/0, S[i].start, S[i].stop);
    long double accCol = 0.0;
    for (int r=S[i].start; r<S[i].stop; ++r){
      long double v = dense[r*mSize + 0];
      accCol += v*v;
    }
    double b = std::sqrt((double)accCol);
    vals.push_back(std::sqrt(a*a + b*b));
  }

  return vals;
}

// ---------- Core runner ----------
static void run_csv(Opts opt){
  const int mSize = opt.maxTypes*opt.maxNumber + 1;

  // default selroot = input with "_selected.root"
  if (opt.selroot.empty()){
    opt.selroot = opt.in;
    if (opt.selroot.size()>=5 && opt.selroot.substr(opt.selroot.size()-5)==".root"){
      opt.selroot = opt.selroot.substr(0,opt.selroot.size()-5) + "_selected.root";
    } else {
      opt.selroot += "_selected.root";
    }
  }

  // default out20 if not provided
  if (opt.out20.empty()){
    opt.out20 = opt.out;
    // insert or append suffix
    if (opt.out20.size()>=4 && opt.out20.substr(opt.out20.size()-4)==".csv"){
      opt.out20 = opt.out20.substr(0,opt.out20.size()-4) + "_compact20.csv";
    } else {
      opt.out20 += "_compact20.csv";
    }
  }

  TFile inFile(opt.in.c_str(),"READ");
  if(!inFile.IsOpen()){ cerr<<"ERROR: cannot open "<<opt.in<<"\n"; return; }
  TTree* ntuple = (TTree*)inFile.Get("Ntuple"); // change if your tree name differs
  if(!ntuple){ cerr<<"ERROR: tree 'Ntuple' not found\n"; return; }

  VecLeaves V;
  if(!bind_schema_vec(ntuple,V)){ cerr<<"ERROR: input tree schema mismatch\n"; return; }

  // --- create selected-events ROOT file & clone schema ---
  TFile outSel(opt.selroot.c_str(),"RECREATE");
  if(!outSel.IsOpen()){ cerr<<"ERROR: cannot create "<<opt.selroot<<"\n"; return; }
  TTree* outTree = ntuple->CloneTree(0);

  // CSV (full RMM)
  ofstream csv(opt.out);
  if(!csv){ cerr<<"ERROR: cannot write "<<opt.out<<"\n"; return; }
  csv.setf(std::ios::fixed); csv<<std::setprecision(6);

  // header (add in_entry for mapping)
  csv<<"run,event,weight,in_entry";
  for (int r=0;r<mSize;r++)
    for (int c=0;c<mSize;c++)
      csv<<",R"<<setw(2)<<setfill('0')<<r<<"C"<<setw(2)<<setfill('0')<<c;
  csv<<"\n";

  // CSV (compact-20)
  ofstream csv20(opt.out20);
  if(!csv20){ cerr<<"ERROR: cannot write "<<opt.out20<<"\n"; return; }
  csv20.setf(std::ios::fixed); csv20<<std::setprecision(6);

  // compact-20 header
  auto labels20 = compact20_labels(opt.maxTypes);
  csv20<<"run,event,weight,in_entry";
  for (const auto& L: labels20) csv20<<","<<L;
  csv20<<"\n";

  Long64_t nent = ntuple->GetEntries();
  if (opt.nevents>0 && opt.nevents<nent) nent = opt.nevents;

  int selected=0;
  for (Long64_t i=0;i<nent;i++){
    ntuple->GetEntry(i);

    vector<LParticle> jets,bjets,mu,el,ph,miss,alljets;
    double met_pt=0.0, met_phi=0.0, wgt=1.0;

    auto fill=[&](const vector<double>* pt,const vector<double>* eta,const vector<double>* phi,
                  const vector<double>* m,double ptMin,double etaMax,vector<LParticle>& out){
      if(!pt||!eta||!phi) return;
      size_t n=min((size_t)opt.maxNumber, pt->size());
      for(size_t k=0;k<n;k++){
        double pT=(*pt)[k], eT=(*eta)[k], pH=(*phi)[k], mass=(m?(*m)[k]:0.0);
        if(ptMin>0 && pT<ptMin) continue;
        if(etaMax>0 && fabs(eT)>etaMax) continue;
        TLorentzVector q; q.SetPtEtaPhiM(pT,eT,pH,mass);
        LParticle lp; lp.SetP(q); out.push_back(lp);
      }
    };
    fill(V.JET_pt,V.JET_eta,V.JET_phi,V.JET_mass,opt.ptJet,opt.etaJet,jets);
    fill(V.bJET_pt,V.bJET_eta,V.bJET_phi,V.bJET_mass,opt.ptJet,opt.etaJet,bjets);
    fill(V.MU_pt,V.MU_eta,V.MU_phi,nullptr,opt.ptLep,opt.etaLep,mu);
    for(auto& x:mu){ TLorentzVector l=x.GetP(); l.SetVectM(l.Vect(),0.10566); x.SetP(l); }
    fill(V.EL_pt,V.EL_eta,V.EL_phi,nullptr,opt.ptLep,opt.etaLep,el);
    for(auto& x:el){ TLorentzVector l=x.GetP(); l.SetVectM(l.Vect(),0.000511); x.SetP(l); }
    fill(V.PH_pt,V.PH_eta,V.PH_phi,nullptr,opt.ptLep,opt.etaLep,ph);
    for(auto& x:ph){ TLorentzVector l=x.GetP(); l.SetVectM(l.Vect(),0.0); x.SetP(l); }

    met_pt = (V.MET_met && !V.MET_met->empty()) ? V.MET_met->at(0) : 0.0;
    met_phi= (V.MET_phi && !V.MET_phi->empty()) ? V.MET_phi->at(0) : 0.0;
    { TLorentzVector m; m.SetPtEtaPhiM(met_pt,0.0,met_phi,0.0); m.SetPz(0);
      LParticle missP; missP.SetP(m); missP.SetType(1); miss.push_back(missP); }
    wgt = (V.Evt_Weight && !V.Evt_Weight->empty()) ? V.Evt_Weight->at(0) : 1.0;

    alljets = jets; alljets.insert(alljets.end(), bjets.begin(), bjets.end());
    sort_by_pt(jets); sort_by_pt(bjets); sort_by_pt(alljets);
    sort_by_pt(mu);   sort_by_pt(el);    sort_by_pt(ph);

    if (!select_event(opt,met_pt,mu,el,alljets)) continue;

    // ---- Event PASSED: write CSV row + fill selected tree ----
    const float CMS = 13000.0f;
    float** A = projectevent(CMS, opt.maxNumber, opt.maxTypes, miss, jets, bjets, mu, el, ph);

    // flatten to dense row-major
    vector<float> dense((size_t)mSize*mSize, 0.0f);
    for (int r=0;r<mSize;r++)
      for (int c=0;c<mSize;c++)
        dense[(size_t)r*mSize + c] = A[r][c];

    for (int x=0;x<mSize;x++) delete [] A[x]; delete [] A;

    int run=0, evt=(++selected);

    // full RMM CSV row
    csv<<run<<","<<evt<<","<<wgt<<","<<i;
    for (auto& d : dense) csv<<","<<d;
    csv<<"\n";

    // compact-20 CSV row
    auto vals20 = compact20_values_from_matrix(dense, opt.maxTypes, opt.maxNumber);
    csv20<<run<<","<<evt<<","<<wgt<<","<<i;
    for (auto v: vals20) csv20<<","<<v;
    csv20<<"\n";

    // fill cloned tree with the current entry's branch contents
    outTree->Fill();
  }

  cout<<"Wrote "<<opt.out<<" (RMM) and "<<opt.out20<<" (compact-20) with "<<selected<<" selected events (processed "<<nent<<")\n";

  // finalize files
  csv.close();
  csv20.close();
  outSel.cd();
  outTree->Write();
  outSel.Close();
  inFile.Close();

  cout<<"Wrote selected events ROOT file: "<<opt.selroot<<endl;
}

// --- Entry 1: positional args ---
void make_RMMs_compact20(const char* in, const char* out, Long64_t nevents=-1,
                         int iconfig=0, int maxN=10, int types=5, const char* out20=nullptr)
{
  Opts opt;
  opt.in=in?in:""; opt.out=out?out:"";
  opt.nevents=nevents; opt.iconfig=iconfig; opt.maxNumber=maxN; opt.maxTypes=types;
  if (out20) opt.out20 = out20;
  if (opt.in.empty()||opt.out.empty()){ cerr<<"Usage: make_RMMs_compact20(\"in.root\",\"out.csv\",N)\n"; return; }
  run_csv(opt);
}

// --- Entry 2: flags via --args ---
void make_RMMs_compact20(){
  int argc = gApplication->Argc(); char** argv = (char**)gApplication->Argv();
  Opts opt; if (!parse_args(argc, argv, opt)) return; run_csv(opt);
}

