# autoGau： Gaussian输入文件生成脚本
autoGau是一款用户给定`mol.xyz`或简单的`mol.gjf`文件，通过选择关键词来生成`mol.gjf`的Python脚本。

相对比Gauss View的图形化界面，autoGau的优点在于提供更多更加常用的关键词选项，包括大部分Gaussian内置或非内置基组与计算方法，对于初学计算的新手来说，autoGau可以通过详细的指导，方便的生成输入文件，在一定程度上可以代替GV的使用。同时，可以搭配脚本，实现批量xyz的转化。
## 依赖
autoGau的一些功能需要调用一定的Python库，首先，请确保自己装有`Python 3.4`以及以上版本，并安装以下的库：
`numpy`与`os`。
`numpy`安装方法：`pip install numpy`

## 使用方法
准备好`mol.xyz`，第一行为原子数，**第二行为电荷与自旋多重度（如果使用guess=fragment，需要把整体与每个片段的电荷与自旋多重度写上）**，第三行之后为坐标，比如：
```
 3
 0 1
 O                  0.00000000    0.00000000    0.12091400
 H                  0.00000000    0.75666800   -0.48365800
 H                  0.00000000   -0.75666800   -0.48365800
```
或者
```
 2
 0 1 0 4 0 -4
 N 0.0 0.0 0.0
 N 0.0 0.0 2.0
```
或者从gv里保存的简单的`mol.gjf`，只需确保有`#`开头的关键词行（无需管关键词是什么）以及**正确的电荷与自旋多重度，且最后有空行**，也可以使用`mol.gjf`做输入文件，此时产生的gjf为`mol_auto.gjf`，`_auto`补充的目的是以原来的名字做区分(ver1.1之后版本)。

`Linux`: 将准备的`mol.xyz`或简单的`mol.gjf`放入`autoGau`文件下，执行`python3 autoGau.py`，根据指示选择关键词，便在当前路径下生成相应的`mol.gjf`。

或者使用`vim ~/.bashrc`命令打开bashrc，键入`G`使光标移到到最后一行开头，键入`shift + 4`使光标移到到行尾，键入`a`，输入回车，之后输入`alias 'Python3 /PATH/autoGau/autoGau.py'`，其中的PATH为程序安装路径，之后键入`ESC`，键入`:wq`退出，在命令行输入`source ~/.bashrc`，便可以在任意目录使用`autoGau`命令启动脚本。

`Win`： 双击`autoGau.py`即可。
## 支持功能
### 计算方法
1. 支持的非内置MPn方法：SCS-MP2，SCSN-MP2，SCS-MP3。
2. 支持自己组合纯密度泛函，如PBE泛函为PBEPBE。
3. 支持用户自定义BLYP/PBEPBE/SLYP式杂化泛函。
4. 支持用户自定义纯泛函的长程修正。
5. 支持多种非内置双杂化泛函，如B2GP-PLYP，DSD-PBEP86-D3(BJ)，revDSD-PBEP86-D3(BJ)。
6. 由于Gaussian多参考速度慢方法少，故不支持CASSCF，GVB等功能。
### 计算任务
1. 支持与g16新关键词`GIC`相关的两个计算任务：平面翻转与多键长同时扫描。
2. 暂时不支持`scan`关键词的部分计算，刚性扫描完全不支持，柔性扫描可以通过`opt`模块通过`ModRedundant`设置。
3. 支持多Link输入文件的产生(ver1.2之后版本)。
### 基组
1. 支持的基组一览：

   STO-3G，Pople系列，D95，D95V，SHC，LanL2(MB/DZ/TZ)，SDD，cc-pVnZ，def，def2，UCBS，DG(DZ/TZ)VP，CBSB7，CEP，MTSmall，MidiX，EPR，pcSseg-n，pcJ-n，x2c，sadlej，SARC-X-DKH2/ZORA，jorge，Sapporo
2. 非内置基组使用文件引入基组的方法。
3. 可以较为方便的使用混合基组与混合赝势基组，无需再自己复制基组信息。
### 其他功能
1. 方便的设置片段波函数初猜，详见`Guess`模块。
2. 方便的设置计算极化率、超极化率关键词与参数。
3. 方便的设置NBO计算的各种关键词。
4. 方便的设置PCM与SMD等溶剂模型的自定义溶剂关键词，内置基组的书写方法可以见`solvents.dat`。
5. 方便的进行BSSE计算。
## 未来计划
1. 完善`scan`关键词模块。
2. 增加模板功能。
3. 丰富关键词（目前版本将不常用、不好用的关键词删除）
4. 增加温度、压力以及频率矫正因子的设置。

## 部分实例
### 使用revTPSS泛函搭配pcSseg-1基组计算甲烷分子的NMR
1. 启动脚本，输入xyz文件，选择4核4GB，并选择1 Link，`CH4.xyz`如下：
    ```
    5
    0 1
    C         -1.34435        0.26533       -0.00000
    H         -0.23495        0.26533        0.00000
    H         -1.71415        0.36099       -1.04157
    H         -1.71415        1.11953        0.60363
    H         -1.71415       -0.68452        0.43794
    ```
2. 选择`DFT`，并选择纯泛函组合，交换与相关泛函都选择`revTPSS`，不需要LC校正与TD，设置D3色散矫正，完成计算方法设置，以上方法的输入顺序为：
   ```
   3 
   b 
   10 
   10 
   n 
   2
   ```
3. 选择`pcSseg-1`基组，无需弥散函数：
   ```
   1 
   18 
   1 
   Enter
   ```
4. 选择`GIAO`方法计算NMR：
   ```
   6
   1
   ```
5. 最后输入9，便完成gjf的生成。
    ```
    %mem = 4GB
    %nproc = 4
    %chk = CH4.chk
    #p RevTPSSRevTPSS em=GD3 nmr=GIAO Gen

    Generated by autoGau.

    0 1
    C     -1.34435000     0.26533000    -0.00000000
    H     -0.23495000     0.26533000     0.00000000
    H     -1.71415000     0.36099000    -1.04157000
    H     -1.71415000     1.11953000     0.60363000
    H     -1.71415000    -0.68452000     0.43794000

    @/home/byt/software/autoGau/basis/pcSseg-1/N


    ```
### 使用MP2配合混合基组，进行N穿越H3平面的计算
1. `NH3.xyz`如下
    ```
    4
    0 1
    H     1.80953700   -0.74056700    0.00001400 
    H     1.80955100    0.66571400    0.81195400 
    H     1.80951800    0.66574100   -0.81197200 
    N     1.41988000    0.19697800    0.00000500
    ```
2. 选择`mp2`方法，并不使用全核(Full)计算：
    ```
    2
    1
    n
    ```
3. 选择混合基组，其中N使用6-31G(d,p)，H使用def2-SVP：
    ```
    2
    2
    N
    2
    4
    (d,p)
    ENTER
    H
    10
    2
    ```
4. 选择使用`GIC`进行计算，构成平面的原子序号为`1 2 3`，N原子移动的步数为20，步长为-0.07：
    ```
    2
    1 3 4
    1
    1 2 3
    20
    -0.07
    4
    ```
5. 最后键入`2 1`关闭对称，生成`NH3.gjf`为：
    ```
    %mem = 4GB
    %nproc = 4
    %chk = NH3.chk
    #p MP2 opt(ModRedundant,GIC) Gen NoSymm

    Generated by autoGau.

    0 1
    H      1.80953700    -0.74056700     0.00001400
    H      1.80955100     0.66571400     0.81195400
    H      1.80951800     0.66574100    -0.81197200
    N      1.41988000     0.19697800     0.00000500

    X1=X(1)
    Y1=Y(1)
    Z1=Z(1)
    X2=X(2)
    Y2=Y(2)
    Z2=Z(2)
    X3=X(3)
    Y3=Y(3)
    Z3=Z(3)
    X4=X(4)
    Y4=Y(4)
    Z4=Z(4)
    NA=(Y2-Y1)*(Z3-Z1)-(Y3-Y1)*(Z2-Z1)
    NB=(Z2-Z1)*(X3-X1)-(Z3-Z1)*(X2-X1)
    NC=(X2-X1)*(Y3-Y1)-(X3-X1)*(Y2-Y2)
    ND=-1.0*(NA*X1+NB*Y1+NC*Z1)
    Dist(NSteps=20,StepSize=-0.07)=(NA*X4+NB*Y4+NC*Z4+ND)/SQRT(NA**2+NB**2+NC**2)

    N 0
    6-31G(d,p)
    ****
    H 0
    def2SVP
    ****


    ```
### 使用TDDFT搭配SMD模型下的自定义溶剂计算乙烷激发态单点能

1. `C2H6.xyz`如下
    ```
    8
    0 1
    C         -2.80255        0.73828       -0.00007
    C         -1.28357        0.71234       -0.00003
    H         -3.18617        0.32139        0.95491
    H         -3.18939        0.12944       -0.84424
    H         -3.16117        1.78329       -0.11092
    H         -0.92494       -0.33267        0.11081
    H         -0.89672        1.32119        0.84414
    H         -0.89996        1.12924       -0.95501
    ```
2. 选择`PBE0`泛函进行TDDFT计算，一共计算8个态：
    ```
    3
    a
    15
    y
    n
    1
    8
    3
    ```
3. 选择`def2-SVP`基组：
    ```
    1
    10
    2
    ```
4. 键入`1`进行单点能计算。
5. 为了展示更多用法，我们此例将初猜设为拓展Huckel方法，溶剂模型为`SMD`，自定义溶剂为`[BMIM][NTf2]`，各种参数如下：
    ```
    eps = 11.5
    eps_inf = 2.0449
    H-Bond Acidity = 0.229
    H-Bond Basicity = 0.265
    Surface Tension At Interface = 61.24
    Carbon Aromaticity = 0.12
    Electronegative Halogenicity = 0.2
    ```
    我们依次输入：
    ```
    4 6
    2
    6 11
    11.5
    2.0449
    0.229
    0.265
    61.24
    0.12
    0.24
    ```
6. 最后的`C2H6.gjf`为：
    ```
    %mem = 4GB
    %nproc = 4
    %chk = C2H6.chk
    #p PBE1PBE TD(Nstates=8) em=GD3BJ def2SVP guess=Huckel scrf(SMD,read,solvent=generic)

    Generated by autoGau.

    0 1
    C     -2.80255000     0.73828000    -0.00007000
    C     -1.28357000     0.71234000    -0.00003000
    H     -3.18617000     0.32139000     0.95491000
    H     -3.18939000     0.12944000    -0.84424000
    H     -3.16117000     1.78329000    -0.11092000
    H     -0.92494000    -0.33267000     0.11081000
    H     -0.89672000     1.32119000     0.84414000
    H     -0.89996000     1.12924000    -0.95501000

    eps=11.5
    epsinf=2.0449
    HBondAcidity=0.229
    HBondBasicity=0.265
    SurfaceTensionAtInterface=61.24
    CarbonAromaticity=0.12
    ElectronegativeHalogenicity=0.24


    ```
### 多Link任务做片段组合波函数单点能计算并验证波函数稳定性
1. 此例使用gjf为输入文件，`N2.gjf`如下;
    ```
    %chk=N2.chk
    # hf/3-21g geom=connectivity

    Title Card Required

    0 1 0 4 0 -4
    N                 -0.24390245   -0.12195122    0.00000000
    N                 -1.33590245   -0.12195122    0.00000000

    1 2 3.0
    2


    ```
    注意，`0 1`代表整体的电荷与自旋多重度，而`0 4 0 -4`则为两个片段（两个N原子）的电荷与自旋多重度。此gjf为gv生成，仅作电荷与自旋多重度的修改。
2. 第一个任务，选择HF/cc-pVDZ计算单点能：
    ```
    2      # 两个Link的任务
    4
    ENTER  # 4核4GB内存
    2
    1
    n
    8
    D
    ENTER  # HF/cc-pVDZ
    1      # 单点能
    2 4
    1      # NoSymm
    9
    2
    1
    2      # 设置片段
    ```
3. 第二个任务，计算级别同上，此时读取第一个任务的`N2.chk`，做波函数稳定性验证：
    ```
    4
    ENTER
    2
    1
    n
    8
    D
    ENTER  # HF/cc-pVDZ
    7
    1      # stable=opt
    2 4
    1
    10     # guess=read
    g09    
    N2.chk # 如果这里选g16，需要使用oldchk的格式
    ```
4. 生成的`N2_auto.gjf`如下：
    ```
    %mem = 4GB
    %nproc = 4
    %chk = N2.chk
    #p HF cc-pVDZ NoSymm guess=fragment=2

    Generated by autoGau.

    0 1 0 4 0 -4
    N(fragment=1)     -0.24390245    -0.12195122     0.00000000
    N(fragment=2)     -1.33590245    -0.12195122     0.00000000

    --Link1--
    %mem = 4GB
    %nproc = 4
    %chk = N2.chk
    #p HF stable=opt cc-pVDZ NoSymm guess=read geom=allcheck


    ```