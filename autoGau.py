import os
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
basis_path = os.path.abspath(os.path.join(script_dir, '.', 'basis'))
sol_path = os.path.abspath(os.path.join(script_dir, '.', 'solvents.dat'))
ECP_basis = ["Aug-cc-pVDZ-PP","Aug-cc-pVTZ-PP","Aug-cc-pVQZ-PP","cc-pVDZ-PP","cc-pVTZ-PP","cc-pVQZ-PP",\
             "LanL2MB","LanL2DZ","LanL2TZ","LanL2TZ(f)",\
             "SDD","SDDAll"]

# read gjf file
# %chk or %mem or %nproc and so an
# #keywords
#
# title
#
# 0 1 <= charge 2S+1
# H 1.00 1.00 1.00
# H 0.00 0.00 0.00
#
# other
def read_gjf(filename):
    atom_date = []
    with open(filename,'r') as file:
        lines = file.readlines()
        start_index = 0
        for index, line in enumerate(lines):
            if line.startswith('#'):
                start_index = index
                break
        start_index += 4 
        iatm = 0
        while start_index < len(lines) and lines[start_index].strip() != '':
            if lines[start_index][0].isdigit() :
                charge_mul = str(lines[start_index])
            else:
                atom_info = lines[start_index].split()
                atom_symbol = atom_info[0]
                x,y,z = map(float,atom_info[1:4])
                atom_date.append((atom_symbol,x,y,z))
                iatm = iatm + 1
            start_index += 1
    return charge_mul,atom_date

# read xyz file
# 2  <= natom
# 0 1 <= charge 2S+1
# H 1.00 1.00 1.00
# H 0.00 0.00 0.00
def read_xyz(filename):
    atom_date = []
    with open(filename,'r') as file:
        natm = int(file.readline())
        charge_mul = str(file.readline())
        iatm = 0
        for line in file:
            atom_info = line.split()
            atom_symbol = atom_info[0]
            x,y,z = map(float,atom_info[1:4])
            atom_date.append((atom_symbol,x,y,z))
            iatm = iatm + 1
            if iatm > natm:
                print("Error: EOF, check the xyz file!")
    return charge_mul,atom_date

#-------------------------------------------------------------------------------
# get the %mem and %nproc
def get_mem_nproc():
    print("\n----------%mem and %nproc----------\n")
    nproc = input("Please enter the shared processors(eg. 24): ")
    mem = input("Please enter the memory(eg. 30GB, if you input ENTER, the memory will be the same as nproc with GB): ")
    if (mem == ''):
        mem = str(nproc) + "GB"
    return nproc,mem
#-------------------------------------------------------------------------------

def get_GIC():
    print("1.Flipping process   2.Scan multiple coordinates simultaneously")
    GIC = int(input())
    if GIC == 1:
        print("Please enter the atomic numbers of three defining plane atoms(SPACE split).")
        input_num = input()
        Nsteps = str(input("Enter the Nsteps: "))
        stepsize = str(input("Enter the StepSize: "))
        matom = str(input("Enter the moving atom number: "))
        atom = list(map(int,input_num.split()))
        GIC_keyw = f"""X1=X({atom[0]})
Y1=Y({atom[0]})
Z1=Z({atom[0]})
X2=X({atom[1]})
Y2=Y({atom[1]})
Z2=Z({atom[1]})
X3=X({atom[2]})
Y3=Y({atom[2]})
Z3=Z({atom[2]})
X4=X({matom})
Y4=Y({matom})
Z4=Z({matom})
NA=(Y2-Y1)*(Z3-Z1)-(Y3-Y1)*(Z2-Z1)
NB=(Z2-Z1)*(X3-X1)-(Z3-Z1)*(X2-X1)
NC=(X2-X1)*(Y3-Y1)-(X3-X1)*(Y2-Y2)
ND=-1.0*(NA*X1+NB*Y1+NC*Z1)
Dist(NSteps={Nsteps},StepSize={stepsize})=(NA*X4+NB*Y4+NC*Z4+ND)/SQRT(NA**2+NB**2+NC**2)
"""
    elif GIC==2:
        n = int(input("Enter the number of scanned variables: "))
        GIC = np.zeros((n,5), dtype=object)
        F = np.zeros(n-1,dtype=object)
        dis = np.zeros((n,1000),dtype=float)
        for i in range(n):
            print(f"Variable {i+1}")
            Nsteps = int(input("Nsteps: "))
            GIC[i,0] = Nsteps
            dt = float(input("StepSize: "))
            GIC[i,1] = dt
            ini = float(input("Initial distance: "))
            GIC[i,2] = ini
            atoms = str(input("Atom number(',' split): "))
            GIC[i,3] = atoms
            name = str(input("Variables' name: "))
            GIC[i,4] = name
            dis[i,0:GIC[i,0]] = [float(GIC[i,2]) + float(j) * GIC[i,1] for j in range(GIC[i,0])]
        for i in range(n):
            if i > 0:
                coefficients = np.polyfit(dis[0,0:GIC[i,0]], dis[i,0:GIC[i,0]], 1)
                coeff = np.round(coefficients, decimals=6)
                F[i-1] = f"{coeff[0]}*{GIC[0,4]}-{GIC[i,4]}"
        GIC_keywr = np.zeros(n, dtype=object)
        for i in range(n):
            if i == 0:
                GIC_keywr[i] = f"{GIC[0,4]}(NSteps={str(GIC[0,0])},StepSize={str(GIC[0,1])})=R({GIC[0,3]})\n"
            else:
                GIC_keywr[i] = f"{GIC[i,4]}=R({GIC[i,3]})\nF{str(i)}(Frozen)={F[i-1]}\n"
        GIC_keyw = ''.join(GIC_keywr)
    return GIC_keyw



#-------------------------------------------------------------------------------
# get job type keywords
def get_opt_keyw():
    print("\n===== opt =====\n")
    opt_dict = {1:"",2:",TS,calcfc,noeigen",3:"ModRedundant",4:"GIC", \
                5:"calcfc",6:"calcall",7:"recalc=",\
                8:"cartesian",9:"Z-Matrix",10:"Redundant",11:"maxstep",12:"maxcycle",
                13:"NoTrust",14:"GDISS",15:"GEDISS",16:"RFO",\
                17:"VeryTight",18:"Tight",19:"Loose",20:"ReadOpt"}
    keynum = []
    print("MUST CHOOSE 1 OR 2!!!!!!")
    print("1.Minimum        2.TS(with calcfc and noigen)")
    print("3.ModRedundant   4.GIC(g16)")
    print("5.calcfc         6.calcall           7.recalc=N")
    print("8.Cartesian      9.Z-Matrix          10.Redundant")
    print("11.maxstep       12.maxcycle         13.NoTrust")
    print("14.GDISS         15.GEDIIS(Default)  16.RFO")
    print("17.VeryTight     18.Tight            19.Loose")
    print("20.ReadOpt")

    opt_input = input("Enter number(SPACE split): ")
    keynum = list(map(int,opt_input.split()))
    if 3 in keynum and 4 not in keynum:
        ModR = []
        opt_other = "MOD"
        n = int(input("Enter the number of ModRedundant: "))
        for i in range(n):
            print(f"Enter NO.{i+1}")
            ModR.append(str(input().rstrip('\n')))
            opt_other = f"{opt_other}{ModR[i]}\n"
    if 4 in keynum:
        opt_other = f"GIC{get_GIC()}"
    elif 3 not in keynum and 4 not in keynum:
        opt_other = "NO"
    if 12 in keynum:
        N = input("Enter the maxcycle :")
        opt_dict[12] = "maxcycle="+str(N)
    if 11 in keynum:
        K = input("Enter the maxstep :")
        opt_dict[11] = "maxstep="+str(K)
    if 7 in keynum:
        M = input("Recalculate force constants every nth point, n = :")
        opt_dict[7] = "recalc=" + str(M)
    if 20 in keynum:
        print("Enter the information you need!")
        print("eg: noatoms atoms=5-70 notatoms=N,O, means only atoms 5-70 are optimized except N, O.")
        info = str(input())
        opt_other = f"read{info}\n"
    if len(keynum) == 1 and keynum[0] != 2:
        job_keyw = 'opt='+opt_dict[keynum[0]]
    if len(keynum) == 1 and keynum[0] == 1:
        job_keyw = 'opt'
    elif len(keynum) ==2 and (1 in keynum) and (2 not in keynum):
        job_keyw = 'opt='+ opt_dict[keynum[0]] + opt_dict[keynum[1]]
    else:
        job_keyw = "opt("
        for num in keynum:
            job_keyw=job_keyw + opt_dict[num] + ','
        job_keyw = job_keyw[:4]+job_keyw[5:-1]
        job_keyw = job_keyw +")"
    return job_keyw,opt_other
    
def get_freq_keyw():
    print("\n===== freq =====\n")
    freq_dict = {1:"raman",2:"VCD",3:"ROA",\
                4:"anharmonic",5:"projected",6:"Numerical",7:"DoubleNumer",8:"Cubic",\
                9:"VibRot",10:"SaveNM",11:"HPModes",12:"IntModes"}
    keynum = []
    print("1.Compute Raman   2.Compute VCD       3.Compute ROA")
    print("4.Anharmonic corrections              5.Compute projected frequencies")
    print("6.Numerical       7.DoubleNumer       8.Cubic")
    print("9.vibrational-rotational              10.Save Normal Modes")
    print("11.HP Modes       12.Internal Modes   13.No more keywords")
    freq_input = input("Enter number(SPACE split): ")
    keynum = list(map(int,freq_input.split()))
    if len(keynum) == 1 and 13 not in keynum:
        job_keyw = 'freq='+freq_dict[keynum[0]]
    elif len(keynum) == 1 and 13 in keynum:
        job_keyw = "freq"
    else:
        job_keyw = "freq("
        for num in keynum:
            job_keyw=job_keyw + freq_dict[num] + ','
        job_keyw = job_keyw[:-1]
        job_keyw = job_keyw +")"
    return job_keyw

def get_IRC_keyw():
    print("\n===== IRC =====\n")
    IRC_dict = {1:"forword",2:"reverse",\
                3:"calcfc",4:"calcall",\
                5:"maxpoint = ",6:"recalc = ",\
                7:"HPC",8:"LQA"}
    print("1.Forward only   2.Reverse only")
    print("3.Calculate once(must choose)   4.Calculate always")
    print("5.Maxpoint       6.recalc(no 4)")
    print("7.HPC            8.LQA")
    keynum = []
    IRC_input = input("Enter number(SPACE split): ")
    keynum = list(map(int,IRC_input.split()))
    if 5 in keynum:
        N = input("Enter the number of compute points:")
        IRC_dict[5] = "maxpoint=" + str(N)
    if 6 in keynum:
        M = input("Recalculate force constants every nth point, n = :")
        IRC_dict[6] = "recalc=" + str(M)
    if len(keynum) == 1:
        job_keyw = 'IRC='+IRC_dict[keynum[0]]
    else:
        job_keyw = "IRC("
        for num in keynum:
            job_keyw=job_keyw + IRC_dict[num] + ','
        job_keyw = job_keyw[:-1]
        job_keyw = job_keyw +")"
    return job_keyw

def get_NMR_keyw():
    print("\n===== NMR =====\n")
    NMR_dict = {1:"GIAO",2:"CSGT",3:"IGAIM",4:"All",\
                5:"spinspin",6:"mixed"}
    print("1.GIAO     2.CSGT     3.IGAIM     4.Compute properties with all methods")
    print("5.Compute spin-spin couplings     6.mixed")
    keynum = []
    NMR_input = input("Enter number(SPACE split): ")
    keynum = list(map(int,NMR_input.split()))
    if len(keynum) == 1:
        job_keyw = 'nmr='+NMR_dict[keynum[0]]
    else:
        job_keyw = "nmr("
        for num in keynum:
            job_keyw = job_keyw + NMR_dict[num] + ','
        job_keyw = job_keyw[:-1]
        job_keyw = job_keyw +")"
    return job_keyw

def get_stable_keyw():
    print("\n===== stable =====\n")
    print("1.Reoptimize the wavefunction   2.No more keywords")
    keynum = int(input())
    if keynum == 1:
        job_keyw = "stable=opt"
    elif keynum == 2:
        job_keyw = "stable"
    return job_keyw

# get the job type
def get_job_type(xyz_name):
    print("\n----------Job Type----------\n")
    print("1.Energy(sp)          2.Optimiztion(opt)")
    print("3.Frequency(freq)     4.opt+freq")
    print("5.IRC                 6.NMR")
    print("7.Stability(stable)   ")
    job_type = input()
    job_keyw = ''
    if job_type == "1":
        job_keyw = ''
        job_other = "NO"
    elif job_type == "2":
        job_keyw,job_other = get_opt_keyw()
    elif job_type == "3":
        job_keyw = get_freq_keyw()
        job_other = "NO"
    elif job_type == "4":
        job_opt_keyw,job_other = get_opt_keyw()
        job_freq_keyw = get_freq_keyw()
        job_keyw = f"{job_opt_keyw} {job_freq_keyw}"
    elif job_type == "5":
        job_keyw = get_IRC_keyw()
        job_other = "NO"
    elif job_type == "6":
        job_keyw = get_NMR_keyw()
        job_other = "NO"
    elif job_type == "7":
        job_keyw = get_stable_keyw()
        job_other = "NO"
    return job_keyw,job_other
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
# get method
def get_semi_keyw():
    print("\n=====Semi-empirical=====\n")
    semi_dict={1:"AM1",2:"PM3",3:"PM3MM",4:"PM6",5:"PM7",\
               6:"PDDG",7:"INDO",8:"CNDO",\
               9:"MNDO",10:"MINDO3",11:"ZIndo"}
    print("1.AM1    2.PM3    3.PM3MM   4.PM6   5.PM7")
    print("6.PDDG   7.INDO   8.CNDO    9.MNDO  10.MINDO3")
    print("11.ZIndo")
    semi_num = int(input())
    if semi_num == 11:
        ZI_dict = {1:"Nstates=",2:"root=",\
                   3:"Singlet",4:"Triplet",5:"50-50"}
        print("1.Nstates=N(default 3)   2.root=i(default 1)   ")
        print("3.Singlet(default)       4.Triplet             5.50-50")
        keynum = []
        ZI_keyw = "=("
        ZI_input = input("Enter number(SPACE split): ")
        keynum = list(map(int,ZI_input.split()))
        if 1 in keynum:
            N = str(input("Nstates = N:"))
            ZI_dict[1] = "Nstates=" + N
        if 2 in keynum:
            i = str(input("root = i: "))
            ZI_dict[2] = "root="+i
        if keynum == []:
            method_keyw = "ZIndo"
        else:
            for num in keynum:
                    ZI_keyw = ZI_keyw+ZI_dict[num]+","
            ZI_keyw = ZI_keyw[:-1]
            method_keyw = "ZIndo"+ZI_keyw +")"
    else:
        method_keyw = semi_dict[semi_num]
    return method_keyw

def get_HF_postHF_keyw():
    print("\n=====HF and post-HF=====\n")
    print("1.Hartree-Fock(HF)                2.Møller-Plesset(MPn)")
    print("3.Configuration Interaction(CI)   4.Coupled Cluster(CC)")
    HF_num = int(input())
    if HF_num == 1:
        print("\n***** HF *****\n")
        TDyn = str(input("TD-HF?(y/n): "))
        if TDyn == "y":
            TDAyn = str(input("Employ the Tamm-Dancoff approximation(TDA)?(y/n): "))
            if TDAyn == "y":
                TD_keyw = " TDA("
            else:
                TD_keyw = " TD("
            TD_dict = {1:"Nstates=",2:"root=",3:"DEmin=",\
                        4:"Singlet",5:"Triplet",6:"50-50"}
            print("1.Nstates=N(default 3)   2.root=i(default 1)   3.DEmin")
            print("4.Singlet(default)       5.Triplet             6.50-50")
            keynum = []
            TD_input = input("Enter number(SPACE split): ")
            keynum = list(map(int,TD_input.split()))
            if 1 in keynum:
                N = str(input("Nstates = N: "))
                TD_dict[1] = "Nstates=" + N
            if 2 in keynum:
                i = str(input("root = i: "))
                TD_dict[2] = "root="+i
            if 3 in keynum:
                evmin = str(input("ev*1000 = "))
                TD_dict[3] = "DEmin="+evmin
            for num in keynum:
                TD_keyw = TD_keyw+TD_dict[num]+","
            TD_keyw = TD_keyw[:-1]
            TD_keyw = TD_keyw +")"
        else: 
            TD_keyw = ""
        method_keyw = f"HF{TD_keyw}" 

    elif HF_num == 2:
        print("\n***** MPn *****\n")
        print("1.MP2       2.MP3        3.MP4(Default MP4(SDTQ))")
        print("4.MP4(DQ)   5.MP4(SDQ)   6.MP5")
        print("7.SCS-MP2   8.SCSN-MP2   9.SCS-MP3")
        MP_dict = {1:"MP2",2:"MP3",3:"MP4",4:"MP4(DQ)",5:"MP4(SDQ)",6:"MP5",\
                   7:"MP2 IOp(3/125=0333312000)",8:"MP2 IOp(3/125=1760000000)",9:"MP3 IOp(3/125=0333312000)"}
        n = int(input())
        MP = [1,2,3,4,5,6]
        if n in MP:
            print("Frozen core(FC) is default, if you need full core(Full), enter y:")
            Full = str(input())
            if Full == "y":
                keys_to_update = list(MP_dict.keys())[:4] + list(MP_dict.keys())[-1:]
                new_value = '(Full)'
                for key in keys_to_update:
                    MP_dict[key] = MP_dict[key]+new_value
                MP_dict[4] = "MP4(DQ,Full)"
                MP_dict[5] = "MP4(SDQ,Full)"
            method_keyw = MP_dict[n]
        else:
            method_keyw = MP_dict[n]
    elif HF_num == 3:
        print("\n***** CI *****\n")
        print("1.CIS   2.CIS(D)  3.CID   4.CISD   5.QCI")
        CI = int(input())
        CI_dict = {1:"CIS",2:"CIS(D)",3:"CID",4:"CISD",5:"QCISD"}
        if CI == 1 or CI == 2:
            CIS_dict = {1:"Nstates=",2:"root=",3:"DEmin=",\
                        4:"Singlet",5:"Triplet",6:"50-50",7:"Full"}
            print("1.Nstates=N(default 3)   2.root=i(default 1)   3.DEmin")
            print("4.Singlet(default)       5.Triplet             6.50-50")
            print("7.Full core(Frozen core(FC) is default)")
            keynum = []
            CIS_keyw = "=("
            CIS_input = input("Enter number(SPACE split): ")
            keynum = list(map(int,CIS_input.split()))
            if 1 in keynum:
                N = str(input("Nstates = N: "))
                CIS_dict[1] = "Nstates=" + N
            if 2 in keynum:
                i = str(input("root = i: "))
                CIS_dict[2] = "root="+i
            if 3 in keynum:
                evmin = str(input("ev*1000 = "))
                CIS_dict[3] = "DEmin="+evmin
            if keynum == []:
                method_keyw = CI_dict[CI]
            else:
                for num in keynum:
                    CIS_keyw = CIS_keyw+CIS_dict[num]+","
                CIS_keyw = CIS_keyw[:-1]
                method_keyw = CI_dict[CI]+CIS_keyw +")"
        elif CI == 3 or CI == 4:
            print("Frozen core(FC) is default, if you need full core(Full), enter y:")
            Full = str(input())
            if Full == "y":
                CID = "(Full)"
            else:
                CID = ""
            method_keyw = CI_dict[CI]+CID
        elif CI == 5:
            QCI_dict = {1:"T",2:"E4T",3:"TQ",3:"T1Diag",4:"Full"}
            print("1.T   2.E4T(must with T)   3.T1Diag   4.Full")
            keynum = []
            QCI_keyw = "=("
            QCI_input = input("Enter number(SPACE split): ")
            keynum = list(map(int,QCI_input.split()))
            if keynum == []:
                method_keyw = CI_dict[CI]
            else:
                for num in keynum:
                    QCI_keyw = QCI_keyw+QCI_dict[num]+","
                QCI_keyw = QCI_keyw[:-1]
                method_keyw = CI_dict[CI]+QCI_keyw +")"
    elif HF_num == 4:
        print("\n***** CC *****\n")
        print("1.CCD       2.CCSD")
        print("3.CCSD(T)   4.CCSD(T1Diag)   5.CCSD(T,T1Diag)")
        print("6.EOM-CCSD  7.Brueckner Doubles(BD)   8.TD(T)")
        CC_dict = {1:"CCD",2:"CCSD",3:"CCSD(T)",4:"CCSD(T1Diag)",5:"CCSD(T,T1Diag)",6:"EOMCCSD",7:"BD",8:"BD(T)"}
        CC = int(input())
        if CC == 6:
            EOM_dict = {1:"Nstates=",2:"root=",3:"NstPIR=",\
                        4:"Singlet",5:"Triplet",6:"EnergyOnly",7:"Full"}
            print("1.Nstates=N(default 3)   2.root=i(default 1)   3.NstPIR=K(default 2)")
            print("4.NCISstate=M            5.Triplet             6.EnergyOnly")
            print("7.Full core(Frozen core(FC) is default)")
            keynum = []
            EOM_keyw = "=("
            EOM_input = input("Enter number(SPACE split): ")
            keynum = list(map(int,EOM_input.split()))
            if 1 in keynum:
                N = str(input("Nstates = N: "))
                EOM_dict[1] = "Nstates=" + N
            if 2 in keynum:
                i = str(input("root = i: "))
                EOM_dict[2] = "root="+i
            if 3 in keynum:
                K = str(input("NstPIR = K: "))
                EOM_dict[3] = "NstPIR="+K
            if 4 in keynum:
                M = str(input("NCISstate = M: "))
                EOM_dict[4] = "NCISstate="+M
            if keynum == []:
                method_keyw = CC_dict[CC]
            else:
                for num in keynum:
                    EOM_keyw = EOM_keyw+EOM_dict[num]+","
                EOM_keyw = EOM_keyw[:-1]
                method_keyw = CC_dict[CC]+EOM_keyw +")"
        else:
            method_keyw = CC_dict[CC]
    return method_keyw

def get_DFT_keyw():
    print("\n===== DFT =====\n")
    Hybrid_dict={1:"B3LYP",2:"B3P86",3:"O3LYP",4:"APFD",5:"wB97XD",\
                 6:"LC-wHPBE",7:"CAM-B3LYP",8:"wB97X",\
                 9:"MN15",10:"M11",11:"PW6B5D3",12:"M08HX",13:"M062X",14:"M052X",\
                 15:"PBE1PBE",16:"HSEH1PBE",17:"OHSE2PBE",18:"OHSE1PBE",19:"PBEh1PBE",\
                 20:"TPSSh",21:"BMK",\
                 22:"HISSbPBE",23:"X3LYP",\
                 24:"BHandH",25:"BHandHLYP"}
    Ex_dict = {1:"S",2:"XA",3:"B",4:"PW91",5:"mPW",6:"G96",7:"PBE",\
               8:"O",9:"TPSS",10:"RevTPSS",11:"BRx",12:"PKZB",13:"wPBEh",14:"PBEh"}
    Corr_dict = {1:"VWN",2:"VWN5",3:"LYP",4:"PL",5:"P86",6:"PW91",7:"B95",\
                 8:"PBE",9:"TPSS",10:"RevTPSS",11:"KCIS",12:"BRC",13:"PKZB"}
    pure_dict = {1:"VSXC",2:"tHCTH",3:"B97D3",4:"M06L",5:"M11L",\
                 6:"MN12L",7:"MN15L",8:"SOGGA11",9:"N12",10:"HCTH"}
    Doub_hyb_dict = {1:"B2PLYP",2:"mPW2PLYP",3:"B2PLYPD3",4:"PBE0DH",5:"PBEQIDH",\
                     6:"B2PLYP IOp(3/125=0360003600,3/76=1000006500,3/77=0350003500,3/78=0640006400)",\
                     7:"DSDPBEP86",\
8:"DSDPBEP86 IOp(3/125=0079905785,3/78=0429604296,3/76=0310006900,3/74=1004) em=GD3BJ IOp(3/174=0437700,3/175=-1,3/176=0,3/177=-1,3/178=5500000)"}
    print("a.Hybrid functions   b.Pure functions(combine)   c.Pure functions(standalone)   d.Double-hybrid")
    DFT_type = input()
    if DFT_type == "a":
        print("1.B3LYP     2.B3P86    3.O3LYP   4.APFD(with dispersion)")
        print("5.wB97XD(with D2 dispersion)     6.Lc-wHPBE   7.CAM-B3LYP")
        print("8.wB97X     9.MN15     10.M11    11.PW6B95D3(with D3)")
        print("12.M08HX    13.M062X   14.M052X")
        print("15.PBE0     16.HSE06   17.HS06   18.HS06(support third derivatives)   19.Hybrid of PBE")
        print("20.TPSSh    21.BMK     22.HISS   23.X3LYP")
        print("24.BHandH   25.BHandHLYP         26.User-Defined")
        hy_num = int(input())
        if hy_num == 26:
            print("P2*EC(HF)+P1(P4*EX(Slater)+P3*DeltaEX(non-local))+P6*EC(local)+P5*DeltaEC(non-local)")
            print("eg. B3LYP: P1P2=1000002000, P3P4=0720008000, P5P6=0810010000.")
            typeud = str(input("BLYP/PBEPBE/SLYP: "))
            IOp76 = str(input("P1 and P2, enter mmmmmnnnnn: "))
            IOp77 = str(input("P3 and P4, enter mmmmmnnnnn: "))
            IOp78 = str(input("P5 and P6, enter mmmmmnnnnn: "))
            DFT_keyw = f"{typeud} IOp(3/76={IOp76},3/77={IOp77},3/78={IOp78})"
        else:
            DFT_keyw = Hybrid_dict[hy_num]
    elif DFT_type == "b":
        print("Combine exchange and correlation functionals.")
        print("Exchange: 1.S    2.XA   3.B      4.PW91       5.mPW    6.G96")
        print("          7.PBE  8.O    9.TPSS   10.RevTPSS   11.BRx   12.PKZB   13.wPBEh   14.PBEh")
        Ex_num = int(input())
        print("Correlation: 1.VWN   2.VWN5   3.LYP    4.PL         5.P86     6.PW91")
        print("             7.B95   8.PBE    9.TPSS   10.RevTPSS   11.KCIS   12.BRC   13.PKZB")
        Corr_num = int(input())
        LCY = str(input("Need Long-range Corrected(LC)?(y/n): "))
        if LCY == "y":
            print("Omega: 1.Default   2.User-Defined")
            LCV = int(input())
            if LCV == 1:
                DFT_keyw = f"LC-{Ex_dict[Ex_num]}{Corr_dict[Corr_num]}"
            else: 
                IOp107 = str(input("IOp(3/107=mmmmmnnnnn): "))
                IOp108 = str(input("IOp(3/108=mmmmmnnnnn): "))
                DFT_keyw = f"LC-{Ex_dict[Ex_num]}{Corr_dict[Corr_num]} IOp(3/107={IOp107},3/108={IOp108})"
        else: 
            DFT_keyw = Ex_dict[Ex_num]+Corr_dict[Corr_num]
    elif DFT_type == "c": 
        print("1.VSXC     2.tHCTH   3.B97D3(with D3)")
        print("4.M06L     5.M11L    6.MN12L   7.NM15L")
        print("8.SOGGA11  9.N12     10.HCTH/*")
        pure_num = int(input())
        if pure_num == 10:
            HCTC = str(input("*=407/93/147: "))
        else:
            HCTC = ""
        DFT_keyw = pure_dict[pure_num]+HCTC
    elif DFT_type == "d":
        print("1.B2PLYP     2.mPW2PLYP   3.B2PLYPD3")
        print("4.PBE0DH     5.PBEQIDH    6.B2GP-PLYP")
        print("7.DSD-PBEP86-D3(BJ)       8.revDSD-PBEP86-D3(BJ)")
        
        Doub_hyb_num = int(input())
        DFT_keyw = Doub_hyb_dict[Doub_hyb_num]
    TDyn = str(input("TD-DFT?(y/n): "))
    if TDyn == "y":
        TDAyn = str(input("Employ the Tamm-Dancoff approximation(TDA)?(y/n): "))
        if TDAyn == "y":
            TD_keyw = " TDA("
        else:
            TD_keyw = " TD("
        TD_dict = {1:"Nstates=",2:"root=",3:"DEmin=",\
                   4:"Singlet",5:"Triplet",6:"50-50"}
        print("1.Nstates=N(default 3)   2.root=i(default 1)   3.DEmin")
        print("4.Singlet(default)       5.Triplet             6.50-50")
        keynum = []
        TD_input = input("Enter number(SPACE split): ")
        keynum = list(map(int,TD_input.split()))
        if 1 in keynum:
            N = str(input("Nstates = N: "))
            TD_dict[1] = "Nstates=" + N
        if 2 in keynum:
            i = str(input("root = i: "))
            TD_dict[2] = "root="+i
        if 3 in keynum:
            evmin = str(input("ev*1000 = "))
            TD_dict[3] = "DEmin="+evmin
        for num in keynum:
            TD_keyw = TD_keyw+TD_dict[num]+","
        TD_keyw = TD_keyw[:-1]
        TD_keyw = TD_keyw +")"
    else: 
        TD_keyw = ""
    DFT_keyw = f"{DFT_keyw} {TD_keyw}" 
    print("Grimme's D3 dispersion: 1.No   2.Zore   3.Becke-Johnson(BJ)")
    em_num = int(input())
    if em_num == 1:
        DFT_keyw = DFT_keyw
    elif em_num == 2:
        DFT_keyw = DFT_keyw + " em=GD3"
    elif em_num == 3:
        DFT_keyw = DFT_keyw + " em=GD3BJ" 
    return DFT_keyw

def get_Gn_CBS_keyw():
    print("\n=====Thermodynamic Combination Method=====\n")
    print("1.Gn   2.CBS   3.W1")
    GCtype = int(input())
    if GCtype == 1:
        print("\n***** Gn *****\n")
        print("1.G3(MP2)   2.G3(MP2)//B3LYP")
        print("3.G4        4.G4(MP2)")
        Gn_dict = {1:"G3MP2",2:"G3MP2B3",3:"G4",4:"G4MP2"}
        Gn = int(input())
        method_keyw = Gn_dict[Gn]
    elif GCtype == 2:
        print("\n***** CBS *****\n")
        print("1.CBS-4M   2.CBS-QB3   3.CBS-APNO")
        CBS_dict = {1:"CBS-4M",2:"CBS-QB3",3:"CBS-APNO"}
        CBS = int(input())
        method_keyw = CBS_dict[CBS]
    elif GCtype == 3:
        print("\n***** W1 *****\n")
        print("1.W1U   2.W1BD   3.W1RO")
        W1_dict = {1:"W1U",2:"W1BD",3:"W1RO"}
        W1 = int(input())
        method_keyw = W1_dict[W1]
    return method_keyw


def get_method_keyw():
    print("\n----------Get the methods----------\n")
    print("1.Semi-empirical    2.HF and post-HF")
    print("3.DFT               4.Thermodynamic Combination Method(Gn/CBS/W1)")
    method_type = int(input())
    if method_type == 1:
        method_keyw = get_semi_keyw()
    elif method_type == 2:
        method_keyw = get_HF_postHF_keyw()
    elif method_type == 3:
        method_keyw = get_DFT_keyw()
    elif method_type == 4:
        method_keyw = get_Gn_CBS_keyw()
    return method_keyw,method_type
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# get basis
def get_basis_keyw():
    print("\n===== Built-in Basis =====\n")
    print("1.STO-3G(H-Xe)       2.Pople(H-Kr)        3.D95(H-Cl except Na, Mg)   4.D95V(H-Ne)")
    print("5.SHC(H-Cl)          6.LanL2(MB/DZ/TZ)    7.SDD(all but Fr and Ra)    8.cc-pVnZ(H-Ar,Ca-Kr)")
    print("9.def(H-Kr)          10.def2(H-La,Hf-Rn)  11.UCBS(H-Lr)               12.DG(DZ/TZ)VP")
    print("13.CBSB7(H-Kr)       14.CEP(H-Rn)         15.MTSmall(H-Ar)            16.MidiX(H,C-F,S-Cl,I,Br)")
    print("17.EPR(H,B,C,N,O,F)  18.pcSseg-n(H-Kr)    19.pcJ-n(H-Ar)              20.x2c(H-Rn)")
    print("21.sadlej            22.SARC-X-DKH2/ZORA  23.jorge(H-Lr)              24.Sapporo(H-Xe,DKH3 for K-Rn)")
    need_zeta = [1,2,6,7,8,9,10,12,14,17,18,19,20,21,22,23,24]
    basis_type = int(input())
    basis_path_keyw = ""
    if basis_type != 25 or basis_type != 26:
        if basis_type in need_zeta:
            if basis_type == 1:
                basis_keyw = "STO-3G"
            elif basis_type == 2:
                print("1.3-21G(H-Xe)   2.6-21G(H-Cl)")
                print("3.4-31G(H-Ne)   4.6-31G(H-Kr)   5.6-311G(H-Kr)")
                Pople = int(input())
                Pople_dict = {1:"3-21",2:"6-21",3:"4-31",4:"6-31",5:"6-311"}
                if Pople == 4 or Pople == 5:
                    print("Polarization Functions: (d/2d/3d/df/2df/3df/3d2f,p/2p/3p/pd/2pd/3pd/3p2d) or (d), no polariztion input ENTER:")
                    polar = str(input())
                    print("Diffuse Functions: +/++, no ENTER:")
                    diff = str(input())
                    basis_keyw = f"{Pople_dict[Pople]}{diff}G{polar}" 
                elif Pople == 1:
                    print("Diffuse Functions: +, no ENTER:")
                    diff = str(input())
                    basis_keyw = f"{Pople_dict[Pople]}{diff}G"
                elif Pople == 2 or Pople == 3:
                    print("Polarization Functions: */**, no polariztion input ENTER:")
                    polar = str(input())
                    basis_keyw = f"{Pople_dict[Pople]}G{polar}"
            elif basis_type == 6:
                print("1.LanL2MB   2.LanL2DZ   3.LanL2TZ    4.LanL2TZ(f)")
                Lanl2_dict = {1:"LanL2MB",2:"LanL2DZ",3:"GenECP",4:"GenECP"}
                Lanl2 = int(input())
                basis_keyw = Lanl2_dict[Lanl2]
                if Lanl2 == 3:
                    basis_path_keyw = f"@{basis_path}/LanL2TZ/N"
                elif Lanl2 == 4:
                    basis_path_keyw = f"@{basis_path}/LanL2TZ(f)/N"
            elif basis_type == 7:
                print("1.SSD   2.ADDall")
                basis_keyw = "SSD"
                SSD_num = int(input())
                if SSD_num == 2:
                    basis_keyw = f"{basis_keyw}all"
            elif basis_type == 8:
                print("n = D/T/Q/5/6(6Z for H,B-Ne) or Pseudo potential(PP) or Douglas-Kroll(DK) or F12 truncation(F12):")
                cc_dict = {"D":"cc-pVDZ","T":"cc-pVTZ","Q":"cc-pVQZ","5":"cc-pV5Z","6":"cc-pV6Z"}
                cc = str(input())
                if cc == "PP":
                    print("n = D/T/Q")
                    cc_PP_dict = {"D":"cc-pVDZ-PP","T":"cc-pVTZ-PP","Q":"cc-pVQZ-PP"}
                    cc_PP = str(input())
                    print("Diffuse Functions: Aug, no ENTER:")
                    cc_PP_diff = str(input())
                    basis_keyw = "GenECP"
                    basis_path_keyw = f"@{basis_path}/{cc_PP_diff}-{cc_PP_dict[cc_PP]}/N"
                elif cc == "DK":
                    print("n = D/T/Q")
                    cc_DK_dict = {"D":"cc-pVDZ-DK","T":"cc-pVTZ-DK","Q":"cc-pVQZ-DK"}
                    cc_DK = str(input())
                    print("Diffuse Functions: Aug, no ENTER:")
                    cc_DK_diff = str(input())
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/{cc_DK_diff}-{cc_DK_dict[cc_DK]}/N"
                elif cc == "F12":
                    print("n = D/T/Q")
                    cc_F12_dict = {"D":"cc-pVDZ-F12","T":"cc-pVTZ-F12","Q":"cc-pVQZ-F12"}
                    cc_F12  = str(input())
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/{cc_F12_dict[cc_F12]}/N"
                else:
                    print("Diffuse Functions: Aug/spAug/dAug/Jul/Jun/May/Apr, no ENTER:")
                    print("Notice: Aug-cc-pVDZ-QZ for H-Ar,Sc-Kr, Aug-cc-pV5Z for H-Na,Al-Ar,Sc-Kr, Aug-cc-pV6Z for H,B-O.")
                    diff = str(input())
                    if diff == '':
                        basis_keyw = cc_dict[cc]
                    else:
                        basis_keyw = f"{diff}-{cc_dict[cc]}"
            elif basis_type == 9:
                print("1.SV   2.SVP   3.TZV   4.TZVP")
                def1_dict = {1:"SV",2:"SVP",3:"TZV",4:"TZVP"}
                def1 = int(input())
                basis_keyw = def1_dict[def1]
            elif basis_type == 10:
                print("1.def2-SV   2.def2-SVP   3.def2-SVPP")
                print("4.def2-TZV  5.def2-TZVP  6.def2-TZVPP")
                print("7.def2-QZV  8.def2-QZVP  9.def2-QZVPP")
                print("10.ma-def2-SVP        11.ma-def2-TZVP      12.ma-def2-TZVPP")
                print("13.ma-def2-TZVP(-f)   14.ma-def2-QZVP      15.ma-def2-QZVPP")
                def2_dict = {1:"def2SV",2:"def2SVP",3:"def2SVPP",\
                             4:"def2TZV",5:"def2TZVP",6:"def2TZVPP",\
                             7:"def2QZV",8:"def2QZVP",9:"def2QZVPP",\
                             10:"ma-SVP",11:"ma-TZVP",12:"ma-TZVPP",\
                             13:"ma-TZVP(-f)",14:"ma-QZVP",15:"ma-QZVPP"}
                ma = [10,11,12,13,14,15]
                def2 = int(input())
                if def2 in ma:
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/{def2_dict[def2]}/N"
                else:
                    basis_keyw = def2_dict[def2]
            elif basis_type == 12:
                print("1.DGDZVP(H-Xe)   2.DGDZVP2(H-F,Al-Ar,Sc-Zn)   3.DGTZVP(H,C-F,Al-Ar)")
                DG_dict = {1:"DGDZVP",2:"DGDZVP2",3:"DGTZVP"}
                DG_num = int(input())
                basis_keyw = DG_dict[DG_num]
            elif basis_type == 14:
                print("1.CEP-4G   2.CEP-31G   3.CEP-121G")
                CEP_dict = {1:"CEP-4G",2:"CEP-31G",3:"CEP-121G"}
                CEP_num = int(input())
                print("Polarization Functions: *(Li-Ar), no polariztion input ENTER:")
                polar = str(input())
                basis_keyw = CEP_dict[CEP_num]+polar
            elif basis_type == 17:
                print("1.EPR-II   2.EPR-III")
                EPR_dict = {1:"EPR-II",2:"EPR-III"}
                EPR_num = int(input())
                basis_keyw = EPR_dict[EPR_num]
            elif basis_type == 18:
                print("n = 0/1/2/3/4:")
                pcS = str(input())
                print("Diffuse Functions: Aug, no ENTER:")
                diff = str(input())+"-"
                if diff == "-":
                    diff = ""
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/{diff}pcSseg-{pcS}/N"
            elif basis_type == 19:
                print("n = 0/1/2/3/4:")
                pcJ = str(input())
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/pcJ-{pcJ}/N"
            elif basis_type == 20:
                print("1.x2c-SV(P)all   2.x2c-SVPall")
                print("3.x2c-TZVPall    4.x2c-TZVPPall")
                print("5.x2c-QZVPall    6.x2c-QZVPPall")
                x2c_dict = {1:"x2c-SV(P)all",2:"x2c-SVPall",\
                            3:"x2c-TZVPall",4:"x2c-TZVPPall",\
                            5:"x2c-QZVPall",6:"x2c-QZVPPall"}
                x2c = int(input())
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/{x2c_dict[x2c]}/N"
            elif basis_type == 21:
                print("1.LPOL-X(H,C-F)   2.POL(H-Fr)   2.ZnPOL(H-Cl)")
                sad = int(input())
                if sad == 1:
                    print("X = dl/ds/fl/fs: ")
                    LPOL = str(input())
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/LPOL-{LPOL}/N"
                elif sad == 2:
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/POL/N"
                elif sad ==3:
                    print("n = 2/3: ")
                    ZPOL = str(input())
                    basis_keyw = "Gen"
                    basis_path_keyw = f"@{basis_path}/Z{ZPOL}POL/N"
            elif basis_type == 22:
                print("SARC-X-DKH2/ZORA")
                print("X = none(La-Lr)/QZV/QZVP(La-Lu): ")
                X = str(input())+"-"
                if X == "none-":
                    X=""
                print("DKH2/ZORA?")
                SARC = str(input())
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/SARC-{X}{SARC}/N"
            elif basis_type == 23:
                print("jorge-nZP-X")
                print("n = D/T(H-Fr)/Q(H-Xe)/5/6(H-Ar)")
                n = str(input())
                print("Need DKH? (y or n): ")
                X = str(input())
                if X == "n":
                    X = ""
                elif X == "y":
                    X = "-DK2"
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/jorge-{n}ZP{X}/N"
            elif basis_type == 24:
                print("Sapporo-DKH3/none-nZP: n = D/T/Q: ")
                n = str(input())
                print("Need DKH3? (y or n): ")
                X = str(input())
                if X == "n":
                    X = ""
                elif X == "y":
                    X = "DKH3-"
                basis_keyw = "Gen"
                basis_path_keyw = f"@{basis_path}/Sapporo-{X}{n}ZP/N"
        else:
            if basis_type == 3:
                print("Polarization Functions: (d/2d/3d/df/2df/3df/3d2f,p/2p/3p/pd/2pd/3pd/3p2d), no polariztion input ENTER:")
                polar = str(input())
                print("Diffuse Functions: +/++, no ENTER:")
                diff = str(input())
                basis_keyw = "D95"+diff+polar
            elif basis_type == 4:
                print("Polarization Functions: (d) or (d,p), no polariztion input ENTER:")
                polar = str(input())
                print("Diffuse Functions: +/++, no ENTER:")
                diff = str(input())
                basis_keyw = "D95V"+diff+polar
            elif basis_type == 5:
                print("Polarization Functions: *, no polariztion input ENTER:")
                polar = str(input())
                basis_keyw = "SHC"+polar
            elif basis_type == 11:
                print("Polarization Functions: 1P/2P/3P, no polariztion input ENTER:")
                polar = str(input())
                print("Diffuse Functions: +/++, no ENTER:")
                diff = str(input())
                basis_keyw = "UGBS"+polar+diff
            elif basis_type == 13:
                print("Diffuse Functions: +/++, no ENTER:")
                diff = str(input())
                basis_keyw = "CBSB7"+diff
            elif basis_type == 15:
                basis_keyw = "MTSmall"
            elif basis_type == 16:
                basis_keyw = "MidiX"
        return basis_keyw,basis_path_keyw


def gen_basis_keyw():
    num = int(input("\nHow many basis will you use? "))
    elem = []
    basis = []
    for _ in range(num):
        input_elem = input("Enter elements(SPACE split): ")
        elem_tuple = tuple(input_elem.split())
        elem.append(elem_tuple)
        basis_keyw,basis_path_keyw = get_basis_keyw()
        if basis_keyw == "Gen" or basis_keyw == "GenECP":
            basis.append(basis_path_keyw)
        elif basis_keyw != "Gen" and basis_keyw != "GenECP":
            basis.append(basis_keyw)
    return elem,basis

def choose_basis():
    print("\n---------- Basis ---------\n")
    basis_keyw = ""
    basis_path_keyw = ""
    elem = []
    basis = []
    print("1.Built-in Basis   2.Gen(mixed)   3.GenECP(mixed)")
    num = int(input())
    if num == 1:
       basis_keyw,basis_path_keyw = get_basis_keyw()
    elif num == 2:
        elem,basis = gen_basis_keyw()
        basis_keyw = "Gen"
    elif num == 3:
        elem,basis = gen_basis_keyw()
        basis_keyw = "GenECP"

    return basis_keyw,basis_path_keyw,elem,basis,num
#-------------------------------------------------------------------------------        

#-------------------------------------------------------------------------------
def get_SCF_keyw():
    print("\n=========================")
    print("The SCF and Int options:")
    print("1.DIIS(Default open)   2.Fermi     3.Vshift=N   4.NoIncFock")
    print("5.Quadratic convergence(QC)        6.XQC")
    print("7.conver=N(Default 8)  8.NoVarAcc  9.Int=Grid   10.acc2e=N(Default 10~G09, 12~G16)")
    SCF_dict = {1:"DIIS",2:"Fermi",3:"Vshift=",4:"NoIncFock",\
                5:"QC",6:"XQC",7:"conver=",8:"NoVarAcc"}
    int_dict = {9:"Grid",10:"acc2e="}
    keynum = []
    SCF_int_keyw_arr=[]
    SCF_type = input("Enter number(SPACE split): ")
    keynum = list(map(int,SCF_type.split()))
    SCF_keyw = "SCF("
    if 1 in keynum:
        print("1.DIIS   2.NoDIIS   3.CDIIS")
        num = int(input())
        DIIS_dict = {1:"DIIS",2:"NoDIIS",3:"CDIIS"}
        SCF_dict[1] = DIIS_dict[num]
    if 3 in keynum:
        N = str(input("Vshift = N(Default 100): "))
        SCF_dict[3] = f"Vshift={N}"
    if 7 in keynum:
        N = str(input("conver = N(10^-N): "))
        SCF_dict[7] = f"conver={N}"
    if 9 in keynum:
        print("Integration grid: 1.Fine(G09 Default)   2.UltraFine(G16 Default)   3.SuperFine")
        int_num = int(input())
        grid_dict = {1:"Fine",2:"UltraFine",3:"SuperFine"}
        int_dict[9] = grid_dict[int_num]
        key = f"{int_dict[9]}"
    if 10 in keynum:
        N = str(input("acc2e = N(10^-N): "))
        int_dict[10] = f"acc2e={N}"
        key = f"{int_dict[10]}"
    if 9 in keynum and 10 in keynum:
        int_keyw = f" int({int_dict[9]},{int_dict[10]})"
    elif 9 not in keynum and 10 not in keynum:
        int_keyw = ""
    else :
        int_keyw = f" int={key}"
    if 9 in keynum:
        keynum = list(filter(lambda x: x not in [9], keynum))
    if 10 in keynum:
        keynum = list(filter(lambda x: x not in [10], keynum))
    for num in keynum:
        if len(keynum)==1:
            SCF_keyw = "SCF="+SCF_dict[num]
        else:
            SCF_keyw = SCF_keyw+SCF_dict[num]+","
    if len(keynum)!=1:
        SCF_keyw = SCF_keyw[:-1]
        SCF_keyw = SCF_keyw +")"
    if keynum == []:
        SCF_int_keyw = int_keyw[1:]
    else:
        SCF_int_keyw = f"{SCF_keyw}{int_keyw}"
    SCF_int_keyw_arr.append(SCF_int_keyw)
    SCF_int_keyw_arr.append("NO")
    return SCF_int_keyw_arr

def get_symm_keyw():
    symm_keyw_arr = []
    print("\n=====================")
    print("The Symmetry options:")
    print("1.NoSymm   2.Loose   3.Tight   4.VeryLoose")
    symm_dict = {1:"NoSymm",2:"symm=Loose",3:"symm=Tight",4:"symm=VeryLoose"}
    symm = int(input())
    symm_keyw = symm_dict[symm]
    symm_keyw_arr.append(symm_keyw)
    symm_keyw_arr.append("NO")
    return symm_keyw_arr

def get_polar_keyw():
    polar_keyw = []
    print("\n==================================================================================================")
    print("Method Capabilities                       Polarizability               Hyperpolarizability")
    print("Analytic 3rd derivatives (HF, most DFT)   Polar (default=Analytic)     Polar (default=Analytic)")
    print("Analytic frequencies (MP2, CIS, ...)      Polar (default=Analytic)     Polar=Cubic")
    print("Only analytic gradients (CCSD, BD, ...)   Polar (default=Numeric)      Polar=DoubleNumer")
    print("No analytic derivatives (CCSD(T), ...)    Polar (default=DoubleNumer)  N/A")
    print("==================================================================================================")
    print("1.Polarizabilities   2.Compute optical rotations")
    print("3.DCSHG(without CPHF=RdFreq)(beta(-2*omega,omega,omega))")
    print("Hyperpolarizability(1st): 4.Analytic    5.Numeric   6.DoubleNumer   7.Cubic")
    print("Hyperpolarizability(2rd): 8.Gamma")
    polar_dict = {1:"polar",2:"polar=OptRot",3:"polar=DCSHG",\
                  4:"polar=Analytic",5:"polar=Numeric",6:"polar=DoubleNumer",7:"polar=Cubic",8:"polar=Gamma"}
    polar = int(input())
    polar_keyw.append(polar_dict[polar])
    ifw = str(input("Frequency?(y/n):"))
    if ifw == "y":
        w = "Wpolar"+str(input("Omega: "))
        if polar != 3:
            polar_keyw[0] = polar_keyw[0]+" CPHF=RdFreq"
        else :
            polar_keyw[0] = polar_keyw[0]
        polar_keyw.append(w)
    else:
        polar_keyw.append("NO")
    return polar_keyw

def get_guess_keyw():
    print("\n==================")
    print("The guess options:")
    print("1.Harris(Default for HF/DFT)   2.Huckel(Default for CNDO, INDO, MNDO, and MINDO3)")
    print("3.AM1   4.INDO   5.core(Default for AM1,PM3, PM3MM, PM6, and PDDG)")
    print("6.mix   7.only   8.always")
    print("9.fragment=N     10.read")
    print("11.save          12.local(Boys)")
    guess_dict = {1:"Harris",2:"Huckel",3:"AM1",4:"INDO",5:"core",6:"mix",
                  7:"only",8:"always",9:"fragment=",10:"read",11:"save",12:"local"}
    keynum = []
    guess_keywr=[0,1,2]
    guess = input("Enter number(SPACE split): ")
    keynum = list(map(int,guess.split()))
    guess_keyw = "guess("
    if 9 in keynum:
        N = str(input("Fragment = N, N: "))
        guess_dict[9] = "fragment="+N 
        atom = ""
        for i in range(int(N)):
            print(f"Fragment {i+1}, enter the atom number(SPACE split): ")
            atom_input = str(input())
            atom = f"{atom},{atom_input}"
        atom = f"Frag{atom[1:]}"
    for num in keynum:
        if len(keynum)==1:
            guess_keyw = "guess="+guess_dict[num]
        else:
            guess_keyw = guess_keyw+guess_dict[num]+","
    if len(keynum)!=1:
        guess_keyw = guess_keyw[:-1]
        guess_keyw = guess_keyw +")"
    guess_keywr[0] = guess_keyw
    if 10 in keynum and 9 not in keynum:
        ver = str(input("G09/G16:"))
        if ver == "G09" or ver == "g09":
            print("chk file: ")
            chk = str(input())
            guess_keywr.append("Chk"+chk)
        elif ver == "G16" or ver == "g16":
            print("oldchk file: ")
            oldchk = str(input())
            guess_keywr[1] = "Old"+oldchk
            guess_keywr[2] = "NO"
    elif 9 in keynum and 10 not in keynum:
        guess_keywr[1] = "NO"
        guess_keywr[2] = str(atom)
    elif 9 in keynum and 10 in keynum:
        ver = str(input("G09/G16:"))
        if ver == "G09" or ver == "g09":
            print("chk file: ")
            chk = str(input())
            guess_keywr.append("Chk"+chk)
        elif ver == "G16" or ver == "g16":
            print("oldchk file: ")
            oldchk = str(input())
            guess_keywr[1] = "Old"+oldchk
            guess_keywr[2] = str(atom)
    if 10 not in keynum and 9 not in keynum:
        guess_keywr[1] = "NO"
        guess_keywr[2] = "NO"
    return guess_keywr

def get_pop_keyw():
    print("\n=======================")
    print("The population options:")
    print("1.none(Default for ZIndo)   2.minimal(Default)   3.regular")
    print("4.full(Default for guess=only)                   5.always")
    print("6.NBO   7.NCS   8.NPA")
    print("9.NBOread   10.saveNBOs   11.saveNLMOs   12.saveMixed")
    print("13.Natural Orbitals   14.Natural Spin Orbitals")
    pop_dict = {1:"none",2:"minimal",3:"regular",4:"full",5:"always",\
                6:"NBO",7:"NCS",8:"NPA",9:"NBOread",10:"saveNBOs",11:"saveNLMOs",12:"saveMixed",\
                13:"NO",14:"NOAB"}
    keynum = []
    pop_keywr=[]
    pop = input("Enter number(SPACE split): ")
    keynum = list(map(int,pop.split()))
    pop_keyw = "pop("
    if 7 in keynum:
        print("diag/all, only NCS enter ENTER:")
        NCS = str(input())
        pop_dict[7] = "NCS"+NCS 
    if 9 in keynum:
        print("Enter the file route: ")
        route = str(input())
        NBOread = f"NBO{route}"
    for num in keynum:
        if len(keynum)==1:
            pop_keyw = "pop="+pop_dict[num]
        else:
            pop_keyw = pop_keyw+pop_dict[num]+","
    if len(keynum)!=1:
        pop_keyw = pop_keyw[:-1]
        pop_keyw = pop_keyw +")"
    pop_keywr.append(pop_keyw)
    if 9 in keynum:
        pop_keywr.append(NBOread)
    else:
        pop_keywr.append("NO")
    return pop_keywr

def get_scrf_keyw():
    print("\n======================")
    print("The solvation options:")
    print("1.PCM(Default)   2.CPCM     3.Dipole")
    print("4.IPCM           5.SCIPCM   6.SMD")
    print("7.Include no-polar(for 1-5)")
    print("8.Externaliteration(for post-HF and TD-DFT)")
    print("9.Built-in solvation")
    print("10.Custom solvent(for 1-5)  11.Custom solvent(for SMD)")
    scrf_dict = {1:"PCM",2:"CPCM",3:"Dipole",4:"IPCM",5:"SCIPCM",6:"SMD",\
                 7:"read",8:"ExternalIteration",\
                 9:"solvent=",10:"read",11:"read,solvent=generic"}
    keynum = []
    scrf_keywr=[]
    scrf = input("Enter number(SPACE split): ")
    keynum = list(map(int,scrf.split()))
    scrf_keyw = "scrf("
    if 7 in keynum:
        nopolar = "DIS\nRep\nCav"
    if 10 in keynum:
        eps = str(input("Solvent static dielectric constant(eps): "))
        epsinf = str(input("Solvent dynamic dielectric constant(epsinf): "))
        eps_inf = f"eps={eps}\nepsinf={epsinf}"
    if 11 in keynum:
        eps = str(input("Solvent static dielectric constant(eps): "))
        epsinf = str(input("Solvent dynamic dielectric constant(epsinf): "))
        HBondAcidity = str(input("H-bond acidity: "))
        HBondBasicity = str(input("H-bond basicity: "))
        SurfaceTensionAtInterface = str(input("Surface tension at interface: "))
        CarbonAromaticity = str(input("Carbon aromaticity: "))
        ElectronegativeHalogenicity = str(input("Electronegative halogenicity: "))
        eps_inf = f"""eps={eps}
epsinf={epsinf}
HBondAcidity={HBondAcidity}
HBondBasicity={HBondBasicity}
SurfaceTensionAtInterface={SurfaceTensionAtInterface}
CarbonAromaticity={CarbonAromaticity}
ElectronegativeHalogenicity={ElectronegativeHalogenicity}"""
    if 9 in keynum:
        print("Enter the solvents' name, ENTER for water, H for help.")
        sol = str(input())
        if sol == "H":
            print(f"See {sol_path}!")
            print("Enter the solvents' name, ENTER for water.")
            sol = str(input())
            if sol == "":
                scrf_dict[9] = "water"
        elif sol == "":
            scrf_dict[9] = "water"
        else:
            scrf_dict[9] = "solvent="+sol 
    for num in keynum:
        if len(keynum)==1 and 1 not in keynum:
            scrf_keyw = "scrf="+scrf_dict[num]
        else:
            scrf_keyw = scrf_keyw+scrf_dict[num]+","
    if len(keynum)!=1:
        if keynum == [1,9] and scrf_dict[9] == "water":
            scrf_keyw = "scrf"
        else:
            scrf_keyw = scrf_keyw[:-1]
            scrf_keyw = scrf_keyw +")"
    else:
        scrf_keyw = "scrf="+scrf_dict[num]
    scrf_keywr.append(scrf_keyw)
    if 7 in keynum:
        scrf_keywr.append(f"Nopolar{nopolar}")
    elif 10 in keynum or 11 in keynum:
        scrf_keywr.append(f"Eps{eps_inf}")
    else:
        scrf_keywr.append("NO")
    return scrf_keywr

def get_density_keyw():
    den_keywr = [0,1]
    print("\n====================")
    print("The density options:")
    print("1.current(Default)   2.all   3.SCF   4.MP2")
    print("5.Transition=N       6.AllTransition")
    print("7.CI   8.CC   9.RhoCI   10.Rho2")
    den_dict = {1:"",2:"=all",3:"=SCF",4:"=MP2",\
                5:"=Transition=N",6:"=AllTransition",\
                7:"=CI",8:"=CC",9:"=RhoCI",10:"=Rho2"}
    den = int(input())
    if den == 5:
        N = str(input("Transition=N, N: "))
        den_dict[5] = "=Transition="+N 
    den_keywr[0] = "density"+den_dict[den]
    den_keywr[1] = "NO"
    return den_keywr

def get_BSSE_keyw():
    print("\n===================")
    print("The BSSE options:")
    N = str(input("Fragment = N, N: "))
    BSSE_keywr = [0,1,2]
    BSSE_keyw = "counterpoise="+N 
    atom = ""
    for i in range(int(N)):
        print(f"Fragment {i+1}, enter the atom number(SPACE split): ")
        atom_input = str(input())
        atom = f"{atom},{atom_input}"
    atom = f"Frag{atom[1:]}"
    BSSE_keywr[0] = BSSE_keyw
    BSSE_keywr[1] = "NO"
    BSSE_keywr[2] = str(atom)
    return BSSE_keywr
    
def none_part():
    non_keywr = ["","NO"]
    return non_keywr

def get_general_keyw():
    print("\n---------- Other ----------\n")
    print("1.SCF and Int   2.Symmetry     3.Polarizabilities")
    print("4.Guess         5.Population   6.Solvation")
    print("7.Density       8.BSSE         9.None")
    general_dict = {1:get_SCF_keyw,2:get_symm_keyw,3:get_polar_keyw,4:get_guess_keyw,\
                    5:get_pop_keyw,6:get_scrf_keyw,7:get_density_keyw,8:get_BSSE_keyw,9:none_part}
    keynum = []
    general_keyw = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    general_type = input("Enter number(SPACE split): ")
    keynum = list(map(int,general_type.split()))
    general_keyw_1 = ""
    for num in keynum:
        keyw = general_dict[num]()
        general_keyw_1 = f"{general_keyw_1} {keyw[0]}"
        if len(keyw) == 2:
            general_keyw[num] = keyw[1]
        if len(keyw) == 3:
            general_keyw[num] = keyw[1]
            general_keyw[14] = keyw[2]
    general_keyw[0] = general_keyw_1[1:]
    return general_keyw


#-------------------------------------------------------------------------------
# generate Gaussian input
def generate_Gau_gjf(file_name,iLink):
    if file_name[-3:] == "gjf":
        char_mul,atom_date = read_gjf(file_name)
        name = file_name.split('.')[0] # get the name without file extension
        gjf_name = f"{name}_auto.gjf"
    elif file_name[-3:] == "xyz":
        char_mul,atom_date = read_xyz(file_name)
        name = file_name.split('.')[0] # get the name without file extension
        gjf_name = f"{name}.gjf"
    chk_name = f"{name}.chk"
    nproc,mem = get_mem_nproc()
    method_keyw,method_type = get_method_keyw()
    if method_type == 4:
        basis_keyw = ""
        basis_path_keyw = ""
        job_keyw = ""
        job_other = "NO"
        elem = []
        basis = []
        basis_num = 0
    elif method_type == 1:
        basis_keyw = ""
        basis_path_keyw = ""
        elem = []
        basis = []
        basis_num = 0
        job_keyw,job_other = get_job_type(file_name)
    elif method_type == 2 or method_type == 3:
        basis_keyw,basis_path_keyw,elem,basis,basis_num = choose_basis()
        job_keyw,job_other = get_job_type(file_name)
    general_keyw = get_general_keyw()
    keywr = [method_keyw,job_keyw,basis_keyw,general_keyw[0]]
    keyw = " ".join(s for s in keywr if s)
    with open(gjf_name,'a') as file:
        if iLink != 0:
            file.write(f"--Link1--\n")
        file.write(f"%mem = {mem}\n")
        file.write(f"%nproc = {nproc}\n")
        if str(general_keyw[4]).startswith("Chk"):
            file.write(f"%chk = {general_keyw[4][3:]}\n")
        elif str(general_keyw[4]).startswith("Old"):
            file.write(f"%oldchk = {general_keyw[4][3:]}\n")
            file.write(f"%chk = {chk_name}\n")
        else:
            file.write(f"%chk = {chk_name}\n")
        if iLink == 0:
            file.write(f"#p {keyw}\n")
        else :
            file.write(f"#p {keyw} geom=allcheck\n")
        if iLink == 0:
            file.write("\nGenerated by autoGau.\n")
            file.write(f"\n{char_mul}")
        if str(general_keyw[14]).startswith("Frag"):
            count = 1
            atom_frag = general_keyw[14][4:]
            for s in atom_frag:
                if s != " ":
                    if s == ",":
                        count = count +1
                    else:
                        n = count
                        tuple_as_list = list(atom_date[int(s)-1])
                        tuple_as_list[0] = f"{tuple_as_list[0]}(fragment={n})"
                        atom_date[int(s)-1] = tuple(tuple_as_list)
        if iLink == 0:
            for atom in atom_date:
                atom_type = atom[0]
                x, y, z = '{:10.8f}'.format(atom[1]), '{:10.8f}'.format(atom[2]), '{:10.8f}'.format(atom[3])
                file.write(f"{atom_type} {x.rjust(15)}{y.rjust(15)}{z.rjust(15)}\n")
        if str(job_other).startswith("read"):
            file.write(f"\n{job_other[4:]}")
        if str(job_other).startswith("GIC"):
            file.write(f"\n{job_other[3:]}")
        elif str(job_other).startswith("MOD"):
            file.write(f"\n{job_other[3:]}")
        if str(general_keyw[3]).startswith("Wpolar"):
            file.write(f"\n{general_keyw[3][6:]}\n") 
        if basis_num == 1:
            if basis_keyw == "Gen" or basis_keyw == "GenECP":
                file.write(f"\n{basis_path_keyw}\n")
        elif basis_num == 2:
            file.write('\n')
            for i in range(len(basis)): 
                file.write(' '.join(map(str, elem[i])) + ' 0\n')
                if str(basis[i]).startswith("@"):
                    basis_path = str(basis[i])[1:-2]
                    with open(f'{basis_path}','r') as basis_file:
                        lines = basis_file.readlines()
                    basis_info = []
                    start = False
                    for line in lines:
                        if start:
                            basis_info.append(line)
                            if  "****" in line:
                                break
                        elif line.startswith(f"-{''.join(elem[i])}     0"):
                            start = True
                    for info in basis_info:
                        file.write(f"{info}")
                else:
                    file.write(f"{basis[i]}\n")
                    file.write("****\n")
        elif basis_num == 3:
            file.write('\n')
            ifECP = []
            for i in range(len(basis)): 
                file.write(' '.join(map(str, elem[i])) + ' 0\n')
                if str(basis[i]).startswith("@"):
                    basis_path = str(basis[i])[1:-2]
                    start_index = max(basis_path.rfind('/'),basis_path.rfind('\\')) + 1
                    end_index = len(basis_path)
                    basis_name = basis_path[start_index:end_index]
                    if basis_name in ECP_basis:
                        ifECP.append(i)
                    with open(f'{basis_path}','r') as basis_file:
                        lines = basis_file.readlines()
                        basis_info = []
                        start = False
                        for line in lines:
                            if start:
                                basis_info.append(line)
                                if  "****" in line:
                                    break
                            elif line.startswith(f"-{''.join(elem[i])}     0"):
                                start = True
                        for info in basis_info:
                            file.write(f"{info}")
                else:
                    basis_name = basis[i]
                    if basis_name in ECP_basis:
                        ifECP.append(i)
                    file.write(f"{basis[i]}\n")
                    file.write("****\n") 
            file.write("\n")
            for i in ifECP:
                if str(basis[i]).startswith("@"):
                    basis_path = str(basis[i])[1:-2]
                    with open(f'{basis_path}','r') as basis_file:
                        lines = basis_file.readlines()
                        ECP_info = []
                        start = False
                        for line in lines:
                            if start:
                                ECP_info.append(line)
                                if  line.startswith("-"):
                                    break
                            elif line.startswith(f"-{(''.join(elem[i])).upper()}     0"):
                                ECP_info.append(line[1:])
                                start = True
                        for info in ECP_info[0:-1]:
                            file.write(f"{info}")
                else:
                    basis_name = basis[i]
                    file.write(' '.join(map(str, elem[i])) + ' 0\n')
                    file.write(f"{basis_name}\n")   
        if str(general_keyw[6]).startswith("Eps"):     
            file.write(f"\n{general_keyw[6][3:]}\n") 
        if str(general_keyw[6]).startswith("Nopolar"):     
            file.write(f"\n{general_keyw[6][7:]}\n") 
        if str(general_keyw[5]).startswith("NBO"):     
            file.write(f"\n$NBO plot file={general_keyw[5][3:]} $END\n") 
        file.write('\n')
    return gjf_name
#-------------------------------------------------------------------------------
        
#-------------------------------------------------------------------------------
# main code
filename = input("Please enter the xyz or gjf file: ")
print("How many jobs(--Link1--) do you want in an input?")
Link = int(input())
for iLink in range(Link):
    gjf_name=generate_Gau_gjf(filename,iLink)
with open(gjf_name,'a') as file:
    file.write(f"\n")
print("Gaussian input file generated succesfully!")
